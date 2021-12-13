import logging
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule

mod = SourceModule("""
__global__ void kernel_setInitialDistanceBasedOnSeed(
			int depth,
			int height,
			int width,
			long ucharsPerRow_seed,
			long ucharsPerSlice_seed,
			unsigned char *devSeed,
			int distChannel,
			long floatsPerRow_dis,
			long floatsPerSlice_dis,
			float *devDis)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
	if (x >= width || y >= height || z >= depth)
    {
        return;
    }
	//Compute offset 
	long srcOffset = z*ucharsPerSlice_seed + y*ucharsPerRow_seed + x;
	long desOffset = z*floatsPerSlice_dis   + y*floatsPerRow_dis + x*distChannel;	
	//Obtain initial read write location
	unsigned char *srcLocation = devSeed;  srcLocation += srcOffset;
	float *desLocation = devDis;  desLocation += desOffset;
	
	unsigned char seed_type = *srcLocation;
	float resVal = seed_type > 0 ? 0.0 : 1.0e10;
	
	//Write in all channel : There is only one channel though for distance map
	for(int j=0; j < distChannel; j++)
	{
		*desLocation = resVal;
		desLocation++;
	}
	return;	
}

__host__ __device__ float4 getImgValAtLocation(long srcOffset, int imgChannel_in, float *tempDevMemory_img)
{
	float4 retVal;
	//Obtain initial read  location
	float *srcLocation = tempDevMemory_img;  srcLocation += srcOffset;
	if(1 == imgChannel_in)
	{ 
		float val1 = *srcLocation; srcLocation++; 
		retVal =  make_float4(val1, 0.0, 0.0, 0.0); 
		return retVal;
	}
	else if(2 == imgChannel_in)
	{
		float val1 = *srcLocation; srcLocation++; float val2 = *srcLocation; srcLocation++; 
		retVal =  make_float4(val1, val2, 0.0, 0.0); 
		return retVal;
	}
	else if(4 == imgChannel_in)
	{
		float val1 = *srcLocation; srcLocation++; float val2 = *srcLocation; srcLocation++; 
		float val3 = *srcLocation; srcLocation++; float val4 = *srcLocation; srcLocation++;
		retVal =  make_float4(val1, val2, val3, val4); 
		return retVal;
	}
	else
	{
		retVal =  make_float4(0.0, 0.0, 0.0, 0.0);
		return retVal;
	}
}
__host__ __device__ float getDistValAtLocation(long srcOffset, float *tempDevMemory_dis)
{
	//Obtain initial read  location
	float *srcLocation = tempDevMemory_dis;  srcLocation += srcOffset;
	return *srcLocation;
}

__host__ __device__ void updateDistValAtLocation(long srcOffset, float *tempDevMemory_dis, float updateVal)
{
	//Obtain initial read  location
	float *srcLocation = tempDevMemory_dis;  srcLocation += srcOffset;
	*srcLocation = updateVal;
	return;
}

__host__ __device__ float get_l2_distance(float4 p_value_v, float4 q_value_v)
{
	float result;
	float4 temp = make_float4(p_value_v.x - q_value_v.x, p_value_v.y - q_value_v.y, p_value_v.z - q_value_v.z, p_value_v.w - q_value_v.w);
	result = sqrt((temp.x * temp.x) + (temp.y * temp.y) + (temp.z * temp.z) + (temp.w + temp.w));
	return result;
}

__host__ __device__ void updateDistAtVoxel(int depth, int height, int width, int x, int y, int z,
	int *dd_dev, int *dh_dev, int *dw_dev, 
    long im_stride_k,  long im_stride_j,  long im_stride_i, int imgChannel_in, long *img_offset_dev,
	long dm_stride_k, long dm_stride_j, long dm_stride_i, int distChannel_in, long *dist_offset_dev,   
	float lambda, float *local_dis_dev, int startInd, int numInd,  float *tempDevMemory_img,  float *tempDevMemory_dis)
{
	//Get (multimodal) image value  at voxel
	long srcOffset_img_voxel = z*im_stride_k + y*im_stride_j + x*im_stride_i;
	float4 p_value_v =  getImgValAtLocation(srcOffset_img_voxel, imgChannel_in, tempDevMemory_img);
	//Get previous distance map value at voxel
	long srcOffset_dis_voxel = z*dm_stride_k + y*dm_stride_j + x*dm_stride_i;
	float p_dis = getDistValAtLocation(srcOffset_dis_voxel, tempDevMemory_dis);
	//Now read the image and distance value of the surrounding subset of 26 voxels according to forward / backwrd strategy
	int read_x, read_y, read_z;
	long srcOffset_img, srcOffset_dis;
	for (int offsetIndex = startInd; offsetIndex < startInd + numInd; offsetIndex++)
	{
		read_z = z+dd_dev[offsetIndex];
		read_y = y+dh_dev[offsetIndex];
		read_x = x+dw_dev[offsetIndex];
		bool outsideRange;
		outsideRange = (read_x < 0 || read_x >= width || read_y < 0 || read_y >= height || read_z < 0 || read_z >= depth);
		if (!outsideRange)
		{
			//Get multimodal image value at surrounding voxel and l2 distance on intensity
			srcOffset_img = srcOffset_img_voxel;
			srcOffset_img += img_offset_dev[offsetIndex];
			float4 q_value_v = getImgValAtLocation(srcOffset_img, imgChannel_in, tempDevMemory_img);
			//float l2dis = length(p_value_v - q_value_v); 
			float l2dis = get_l2_distance(p_value_v, q_value_v);
			//Get (previous) distance map at surrounding voxel
			srcOffset_dis = srcOffset_dis_voxel;
			srcOffset_dis += dist_offset_dev[offsetIndex];
			float q_dis = getDistValAtLocation(srcOffset_dis, tempDevMemory_dis);
			//#https://github.com/taigw/GeodisTK/issues/11
			//float speed = (1.0 - lambda) + lambda/(l2dis + 1e-5);
			float speed = (1.0 - lambda) + lambda* local_dis_dev[offsetIndex]/(l2dis + 1e-5);
			float delta_d = local_dis_dev[offsetIndex] / speed;
			float temp_dis = q_dis + delta_d;
			if(temp_dis < p_dis) 
				p_dis = temp_dis;
		}
	}
	//Update distance map
	updateDistValAtLocation(srcOffset_dis_voxel, tempDevMemory_dis, p_dis);	
	return;
}

__global__ void kernel_updateDistanceMap_Fwd_k(int depth, int height, int width, 
	int *dd_dev, int *dh_dev, int *dw_dev, 
    long im_stride_k,  long im_stride_j,  long im_stride_i, int imgChannel_in, long *img_offset_dev,
	long dm_stride_k, long dm_stride_j, long dm_stride_i, int distChannel_in, long *dist_offset_dev,   
	float lambda, float *local_dis_dev, int startInd, int numInd, int updateSliceInd, float *tempDevMemory_img,  float *tempDevMemory_dis)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = updateSliceInd;
	if (x >= width || y >= height || z >= depth)
    {
        return;
    }
	updateDistAtVoxel(depth, height, width, x, y, z,
		dd_dev, dh_dev, dw_dev, 
		im_stride_k, im_stride_j, im_stride_i, imgChannel_in,  img_offset_dev,
		dm_stride_k, dm_stride_j, dm_stride_i, distChannel_in, dist_offset_dev,   
		lambda, local_dis_dev, startInd, numInd, tempDevMemory_img,  tempDevMemory_dis);		
	return;
}

__global__ void kernel_updateDistanceMap_Bwd_k(int depth, int height, int width, 
	int *dd_dev, int *dh_dev, int *dw_dev, 
    long im_stride_k,  long im_stride_j,  long im_stride_i, int imgChannel_in, long *img_offset_dev,
	long dm_stride_k, long dm_stride_j, long dm_stride_i, int distChannel_in, long *dist_offset_dev,
	float lambda, float *local_dis_dev, int startInd, int numInd, int updateSliceInd, float *tempDevMemory_img,  float *tempDevMemory_dis)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = updateSliceInd;
	if (x >= width || y >= height || z >= depth)
    {
        return;
    }
	updateDistAtVoxel(depth, height, width, x, y, z,
		dd_dev, dh_dev, dw_dev, 
		im_stride_k, im_stride_j, im_stride_i, imgChannel_in,  img_offset_dev,
		dm_stride_k, dm_stride_j, dm_stride_i, distChannel_in, dist_offset_dev,   
		lambda, local_dis_dev, startInd, numInd, tempDevMemory_img,  tempDevMemory_dis);		
	return;
}

__global__ void kernel_updateDistanceMap_Fwd_j(int depth, int height, int width, 
	int *dd_dev, int *dh_dev, int *dw_dev, 
    long im_stride_k,  long im_stride_j,  long im_stride_i, int imgChannel_in, long *img_offset_dev,
	long dm_stride_k, long dm_stride_j, long dm_stride_i, int distChannel_in, long *dist_offset_dev,
	float lambda, float *local_dis_dev, int startInd, int numInd, int updateCoronalInd, float *tempDevMemory_img,  float *tempDevMemory_dis)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = updateCoronalInd;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
	if (x >= width || y >= height || z >= depth)
    {
        return;
    }
	updateDistAtVoxel(depth, height, width, x, y, z,
		dd_dev, dh_dev, dw_dev, 
		im_stride_k, im_stride_j, im_stride_i, imgChannel_in,  img_offset_dev,
		dm_stride_k, dm_stride_j, dm_stride_i, distChannel_in, dist_offset_dev,   
		lambda, local_dis_dev, startInd, numInd, tempDevMemory_img,  tempDevMemory_dis);		
	return;
}

__global__ void kernel_updateDistanceMap_Bwd_j(int depth, int height, int width, 
	int *dd_dev, int *dh_dev, int *dw_dev, 
    long im_stride_k,  long im_stride_j,  long im_stride_i, int imgChannel_in, long *img_offset_dev,
	long dm_stride_k, long dm_stride_j, long dm_stride_i, int distChannel_in, long *dist_offset_dev,
	float lambda, float *local_dis_dev, int startInd, int numInd, int updateCoronalInd, float *tempDevMemory_img,  float *tempDevMemory_dis)
{
	int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = updateCoronalInd;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
	if (x >= width || y >= height || z >= depth)
    {
        return;
    }
	updateDistAtVoxel(depth, height, width, x, y, z,
		dd_dev, dh_dev, dw_dev, 
		im_stride_k, im_stride_j, im_stride_i, imgChannel_in,  img_offset_dev,
		dm_stride_k, dm_stride_j, dm_stride_i, distChannel_in, dist_offset_dev,   
		lambda, local_dis_dev, startInd, numInd, tempDevMemory_img,  tempDevMemory_dis);		
	return;
}

__global__ void kernel_updateDistanceMap_Fwd_i(int depth, int height, int width, 
	int *dd_dev, int *dh_dev, int *dw_dev, 
    long im_stride_k,  long im_stride_j,  long im_stride_i, int imgChannel_in, long *img_offset_dev,
	long dm_stride_k, long dm_stride_j, long dm_stride_i, int distChannel_in, long *dist_offset_dev,
	float lambda, float *local_dis_dev, int startInd, int numInd, int updateSagittalInd, float *tempDevMemory_img,  float *tempDevMemory_dis)
{
	int x = updateSagittalInd;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
	if (x >= width || y >= height || z >= depth)
    {
        return;
    }
	updateDistAtVoxel(depth, height, width, x, y, z,
		dd_dev, dh_dev, dw_dev, 
		im_stride_k, im_stride_j, im_stride_i, imgChannel_in,  img_offset_dev,
		dm_stride_k, dm_stride_j, dm_stride_i, distChannel_in, dist_offset_dev,   
		lambda, local_dis_dev, startInd, numInd, tempDevMemory_img,  tempDevMemory_dis);		
	return;
}

__global__ void kernel_updateDistanceMap_Bwd_i(int depth, int height, int width, 
	int *dd_dev, int *dh_dev, int *dw_dev, 
    long im_stride_k,  long im_stride_j,  long im_stride_i, int imgChannel_in, long *img_offset_dev,
	long dm_stride_k, long dm_stride_j, long dm_stride_i, int distChannel_in, long *dist_offset_dev,
	float lambda, float *local_dis_dev, int startInd, int numInd, int updateSagittalInd, float *tempDevMemory_img,  float *tempDevMemory_dis)
{
	int x = updateSagittalInd;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int z = blockIdx.z*blockDim.z + threadIdx.z;
	if (x >= width || y >= height || z >= depth)
    {
        return;
    }
	updateDistAtVoxel(depth, height, width, x, y, z,
		dd_dev, dh_dev, dw_dev, 
		im_stride_k, im_stride_j, im_stride_i, imgChannel_in,  img_offset_dev,
		dm_stride_k, dm_stride_j, dm_stride_i, distChannel_in, dist_offset_dev,   
		lambda, local_dis_dev, startInd, numInd, tempDevMemory_img,  tempDevMemory_dis);		
	return;
}
""")

#Get list of kernel functions
kernel_setInitialDistanceBasedOnSeed = mod.get_function("kernel_setInitialDistanceBasedOnSeed")
kernel_updateDistanceMap_Fwd_k = mod.get_function("kernel_updateDistanceMap_Fwd_k")
kernel_updateDistanceMap_Bwd_k = mod.get_function("kernel_updateDistanceMap_Bwd_k")
kernel_updateDistanceMap_Fwd_j = mod.get_function("kernel_updateDistanceMap_Fwd_j")
kernel_updateDistanceMap_Bwd_j = mod.get_function("kernel_updateDistanceMap_Bwd_j")
kernel_updateDistanceMap_Fwd_i = mod.get_function("kernel_updateDistanceMap_Fwd_i")
kernel_updateDistanceMap_Bwd_i = mod.get_function("kernel_updateDistanceMap_Bwd_i")

def gpuGeodesic3d_raster_scan(I, clickSegmentation, spacing, lamb, iter):
	"""
	I : 3D mulichannel image organized <depth, height, width, channel>:  numpy  ndarray of dtype np.float32 
	clickSegmentation: 	3D binary seed organized <depth, height, width>: numpy  ndarray of dtype np.uint8 
	spacing: list of ('float' valued) pixel spacing (has to be same for both channel) along depth, height and width channel
	lamb: 'float' value of lambda between 0.0 and 1.0; 0.0 => Full Eucliden, 1.0 => Full geodesic
	iter: 'int' value number of iteration of raster scan
	"""
	#
	#  * Wrapper function for the CUDA kernel function.
	#  * @param img multichannel float valued input CT-PET volume: img.
	#  * @param seeds single channel unsigned char valued input binary mask volume: seeds).
	#  * @param distance single channel  float valued output distance map volume: dist.
	#  * @param depth int valued depth of the volume(s)
	#  * @param height int valued height of the volume(s)
	#  * @param width int valued width of the volume(s)
	#  * @param channel int valued number of channels  of the img volume
	#  * @param spacing float  vector (size=3) of spacing along depth, height, width direction
	#  * @param lambda float valued mix (between 0.0 and 1.0) of Euclidean and Geodesic distance: 0 => full Euclidean, 1 => full Geodesic
	#  * @param iter int valued iteration number
	#  **/
	# void GpuGeodisKernel(const float * img, const unsigned char * seeds, float * distance,
	#                             int depth, int height, int width, int channel, 
	#                             std::vector<float> spacing, float lambda, int iteration) 
	# 

	#Result between GeodisTk (CPU Geodesic) and this GPU version showing difference associated with axis switch if I leave like this
	#So attempting axis match
	#img = I.astype(np.float32)
	#seeds = clickSegmentation.astype(np.uint8)
	img = np.transpose(I.astype(np.float32), (2,1,0,3)).copy()
	seeds = np.transpose(clickSegmentation.astype(np.uint8), (2,1,0)).copy()

	assert img.shape[0:3] == seeds.shape, f"Mismatched dimension in first 3 dimensions of img and seeds"
	distance = np.zeros_like(seeds, dtype=np.float32)
	#Result between GeodisTk (CPU Geodesic) and this GPU version showing difference associated with axis switch if I leave like this
	#So attempting axis match
	distance_out = np.transpose(distance, (2,1,0)).copy()

	depth = img.shape[0] #np.intc(img.shape[0]) #
	height = img.shape[1] #np.intc(img.shape[1]) #
	width = img.shape[2] #np.intc(img.shape[2]) #
	channel = img.shape[3] #np.intc(img.shape[3]) #

	#Need to do axis match here too as both img and seed axis have been switched
	spacing = np.array([spacing[2], spacing[1], spacing[0]],dtype=np.float32)
	

	# lamb = np.single(lamb)
	# iteration = np.intc(iter)

	# 	// Compute various pitches / strides to navigate single and multichannel data (seed, image, distance map)
	# 	// Acronyms: dm : distance map; im : image; sm: seed map; stride : basePitch; 
	# 	// byte size per voxel per channel for image and distance = 4, since float valued voxel per channel 
	# 	// C (num channel in input image)
	#   // stride_k = H*W; stride_j = W; stride_i = 1; 
	# 	// sm_stride_k = H*W;   sm_stride_j = W;   sm_stride_i = 1;  Seedmap      is single channel, uChar valued
	# 	// im_stride_k = H*W*C; im_stride_j = W*C; im_stride_i = C;  Image        is multi  channel, float valued
	# 	// dm_stride_k = H*W;   dm_stride_j = W;   dm_stride_i = 1;  Distance map is single channel  float valued	
	# 	int imgChannel_in = channel;
	# 	int distChannel_in = 1;
	# 	long stride_k = height*width; long stride_j = width; long  stride_i = 1;
	# 	long sm_stride_k = stride_k;                long sm_stride_j = stride_j;                long sm_stride_i = stride_i;
	# 	long im_stride_k = stride_k*imgChannel_in;  long im_stride_j = stride_j*imgChannel_in;  long im_stride_i = stride_i*imgChannel_in;
	# 	long dm_stride_k = stride_k*distChannel_in; long dm_stride_j = stride_j*distChannel_in; long dm_stride_i = stride_i*distChannel_in;
	imgChannel_in = channel #np.intc(channel) #
	distChannel_in = 1 #np.intc(1) #

	stride_k = height*width #np.int_(height*width) #
	stride_j = width #np.int_(width) #
	stride_i = 1 #np.int_(1) #

	sm_stride_k = stride_k #np.int_(stride_k) #
	sm_stride_j = stride_j #np.int_(stride_j) #
	sm_stride_i = stride_i #np.int_(stride_i) #

	im_stride_k = stride_k*imgChannel_in #np.int_(stride_k*imgChannel_in) #
	im_stride_j = stride_j*imgChannel_in #np.int_(stride_j*imgChannel_in) #
	im_stride_i = stride_i*imgChannel_in #np.int_(stride_i*imgChannel_in) #

	dm_stride_k = stride_k*distChannel_in #np.int_(stride_k*distChannel_in) #
	dm_stride_j = stride_j*distChannel_in #np.int_(stride_j*distChannel_in) #
	dm_stride_i = stride_i*distChannel_in #np.int_(stride_i*distChannel_in) #

	# 	float s0Sqr = spacing[0] *spacing[0];
	#   float s1Sqr = spacing[1] *spacing[1];
	#   float s2Sqr = spacing[2] *spacing[2];
	s0Sqr = np.single(spacing[0] * spacing[0])
	s1Sqr = np.single(spacing[1] * spacing[1])
	s2Sqr = np.single(spacing[2] * spacing[2])

	# 	//Local distances for 26-neighbourhood of a voxel are now computed and ARRANGED according the way the update kernels will access them
	# 	// Accessed by         <--Center--> <-----Update  k-forward-----------> <-----Update k-backward------------>  <--Ud j-Dwn --> <--Ud j-Up--> <-- Ud i-R --> <-- Ud i-L -->  
	# 	//             INDEX    0            1   2   3   4   5   6   7   8   9    10  11  12  13  14  15  16  17  18    19   20   21   22   23  24        25           26 
	# 	// Read from depth(k)   0,          -1, -1, -1, -1, -1, -1, -1, -1, -1,   +1, +1, +1, +1, +1, +1, +1, +1, +1,    0,   0,   0,   0,   0,  0,        0,           0
	# 	// Read from height(j)  0,          -1, -1, -1,  0,  0,  0, +1, +1, +1,   -1, -1, -1,  0,  0,  0, +1, +1, +1,   -1,  -1,  -1,  +1,  +1, +1,        0,           0
	# 	// Read from width(i)   0,          -1,  0, +1, -1,  0, +1, -1,  0, +1,   -1,  0, +1, -1,  0, +1, -1,  0, +1,   -1,   0,  +1,  -1,   0, +1,       -1,          +1
	# 	// int lookupId_Fwd_k[9] = {1,2,3,4,5,6,7,8,9};
	# 	// int lookupId_Bwd_k[9] = {10,11,12,13,14,15,16,17,18};
	# 	// int lookupId_Fwd_j[3] = {19,20,21};
	# 	// int lookupId_Bwd_j[3] = {22,23,24};
	# 	// int lookupId_Fwd_i[1] = {25};
	# 	// int lookupId_Bwd_i[1] = {26};
	dd_ = np.intc([0, -1, -1, -1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1, +1,  0,  0,  0,  0,  0,  0,  0,  0])
	dh_ = np.intc([0, -1, -1, -1,  0,  0,  0, +1, +1, +1, -1, -1, -1,  0,  0,  0, +1, +1, +1, -1, -1, -1, +1, +1, +1,  0,  0])
	dw_ = np.intc([0, -1,  0, +1, -1,  0, +1, -1,  0, +1, -1,  0, +1, -1,  0, +1, -1,  0, +1, -1,  0, +1, -1,  0, +1, -1, +1])	
	dist_offset_ = np.int_([dd_[i] * dm_stride_k + dh_[i] * dm_stride_j + dw_[i] * dm_stride_i for i in range(27)])
	img_offset_ = np.int_([dd_[i] * im_stride_k + dh_[i] * im_stride_j  + dw_[i] * im_stride_i for i in range(27)])
	local_dis_ = np.single([np.sqrt(abs(dd_[i])*s0Sqr + abs(dh_[i])*s1Sqr + abs(dw_[i])*s2Sqr) for i in range(27)])

	#kernel_name[(griddimx, griddimy, griddimz), (blockdimx, blockdimy, blockdimz)](arguments)
	# 	//Assign threads per blocks (a.k.a BlockDim) and GridDim  (a.k.a GridDim) for launching kernel
	a_BlockDim_vol=(8, 8, 8) 
	a_GridDim_vol=(width // a_BlockDim_vol[0], height // a_BlockDim_vol[1], depth // a_BlockDim_vol[2])
	
	a_BlockDim_fixSlice = (8, 8, 1)
	a_GridDim_fixSlice = (width // a_BlockDim_fixSlice[0], height // a_BlockDim_fixSlice[1], 1)
	
	a_BlockDim_fixCoronal = (8,1,8)
	a_GridDim_fixCoronal = (width // a_BlockDim_fixCoronal[0], 1, depth // a_BlockDim_fixCoronal[2])
	
	a_BlockDim_fixSagittal = (1, 8, 8)
	a_GridDim_fixSagittal = (1, height // a_BlockDim_fixSagittal[1], depth // a_BlockDim_fixSagittal[2])

	#Debug:
	# print(f"local_dis_ : {local_dis_}")
	# print(f"a_BlockDim_vol : {a_BlockDim_vol}")
	# print(f"a_GridDim_vol : {a_GridDim_vol}")
	# print(f"a_BlockDim_fixSlice : {a_BlockDim_fixSlice}")
	# print(f"a_GridDim_fixSlice : {a_GridDim_fixSlice}")
	# print(f"a_BlockDim_fixCoronal : {a_BlockDim_fixCoronal}")
	# print(f"a_GridDim_fixCoronal : {a_GridDim_fixCoronal}")
	# print(f"a_BlockDim_fixSagittal : {a_BlockDim_fixSagittal}")
	# print(f"a_GridDim_fixSagittal : {a_GridDim_fixSagittal}")

	errorHappened = False

	#parameters for different kernel calls for which memory is  NOT allocated on device
	param_depth = np.intc(depth)# int depth,
	param_height = np.intc(height)# int height,
	param_width = np.intc(width)# int width,

	param_imgChannel_in = np.intc(imgChannel_in) #int imgChannel_in
	param_distChannel_in = np.intc(distChannel_in)# int distChannel_in,

	param_sm_stride_k = np.int_(sm_stride_k)# long ucharsPerSlice_seed, long sm_stride_k
	param_sm_stride_j = np.int_(sm_stride_j)# long ucharsPerRow_seed, long sm_stride_j
	param_sm_stride_i = np.int_(sm_stride_i)# long ucharsPerVoxel_seed, long sm_stride_i
	
	param_im_stride_k = np.int_(im_stride_k)# long im_stride_k
	param_im_stride_j = np.int_(im_stride_j)# long im_stride_j
	param_im_stride_i = np.int_(im_stride_i)# long im_stride_j
	
	param_dm_stride_k = np.int_(dm_stride_k)# long floatsPerSlice_dis,	long dm_stride_k
	param_dm_stride_j = np.int_(dm_stride_j)# long floatsPerRow_dis, long dm_stride_j
	param_dm_stride_i = np.int_(dm_stride_i)# long floatsPerVoxel_dis, long dm_stride_i

	param_lambda = np.single(lamb)# float lambda
	
	#parameters for different kernel calls for which memory is ALLOCATED on device
	
	# 	//Create device memory to hold dd_ and transfer dd_ from host to device
	param_dd_dev = cuda.mem_alloc(dd_.nbytes)# int *dd_dev
	cuda.memcpy_htod(param_dd_dev, dd_)
	# //Create device memory to hold dh_ and transfer dh_ from host to device
	param_dh_dev = cuda.mem_alloc(dh_.nbytes)# int *dh_dev
	cuda.memcpy_htod(param_dh_dev, dh_)
	# //Create device memory to hold dw_ and transfer dw_ from host to device
	param_dw_dev = cuda.mem_alloc(dw_.nbytes)# int *dw_dev
	cuda.memcpy_htod(param_dw_dev, dw_)
	# //Create device memory to hold dist_offset and transfer dist_offset from host to device
	param_dist_offset_dev = cuda.mem_alloc(dist_offset_.nbytes)# long *dist_offset_dev
	cuda.memcpy_htod(param_dist_offset_dev, dist_offset_)
	# //Create device memory to hold img_offset_ and transfer img_offset_ from host to device
	param_img_offset_dev = cuda.mem_alloc(img_offset_.nbytes)# long *img_offset_dev
	cuda.memcpy_htod(param_img_offset_dev, img_offset_)
	# //Create device memory to hold local_dis_ and transfer local_dis_ from host to device
	param_local_dis_dev = cuda.mem_alloc(local_dis_.nbytes) # float *local_dis_dev
	cuda.memcpy_htod(param_local_dis_dev, local_dis_)
	# //Create device memory to hold seeds and transfer seeds from host to device	
	param_tempDevMemory_seed = cuda.mem_alloc(seeds.nbytes)# unsigned char *devSeed
	# #For testing
	# clickLocs = [(70,70,67), (57, 54, 62), (81, 90, 99)]
	# unClickLocs = [(70,70,63), (57, 57, 62), (81, 89, 99)]
	# for click in clickLocs:
	# 	seeds[click] = 1
	cuda.memcpy_htod(param_tempDevMemory_seed, seeds)
	# //Create device memory to hold img and transfer img from host to device
	param_tempDevMemory_img = cuda.mem_alloc(img.nbytes)# float *tempDevMemory_img
	#https://stackoverflow.com/questions/62404832/why-cuda-to-device-variable-transfer-gives-noncontiguous-error
	#https://stackoverflow.com/questions/67304253/how-can-i-copy-a-part-of-4d-array-from-host-memory-to-device-memory
	#https://stackoverflow.com/questions/26998223/#what-is-the-difference-between-contiguous-and-non-contiguous-arrays#:~:text=A%20contiguous%20array%20is%20just,reshape(3%2C4)%20.
	#cuda.memcpy_htod(param_tempDevMemory_img, img)	<--- giving non contigous error for img
	############ All CUDA errors are automatically translated into Python exceptions. ####################
	try:
		img_contiguous = np.ascontiguousarray(img)
		cuda.memcpy_htod(param_tempDevMemory_img, img_contiguous)
	#https://realpython.com/the-most-diabolical-python-antipattern/
	except Exception as ex:
		errorHappened = True
		print("uda.memcpy_htod img --> param_tempDevMemory_img failed")
		logging.exception('Caught an error')		
	finally:
		if(True == errorHappened):
			print("Returning with error")
			return distance_out

	# //Create device memory to hold distance and initialze
	param_tempDevMemory_dis = cuda.mem_alloc(distance.nbytes)# float *devDis, float *tempDevMemory_dis
	cuda.memcpy_htod(param_tempDevMemory_dis, distance)
	
	#printf("Initialize distance map using seed location.\n:")
	#print("Initialize distance map using seed location.")	
	try:
		kernel_setInitialDistanceBasedOnSeed(
			param_depth, param_height, param_width, 
			param_sm_stride_j, param_sm_stride_k, param_tempDevMemory_seed,
			param_distChannel_in,param_dm_stride_j, param_dm_stride_k, param_tempDevMemory_dis,
			grid=a_GridDim_vol, block=a_BlockDim_vol)
	except Exception as ex:
		errorHappened = True
		print("kernel_setInitialDistanceBasedOnSeed launch failed")
		logging.exception('Caught an error')
	finally:
		if(True == errorHappened):
			print("Returning with error")
			return distance_out

	#Now iterate:	
	for it in range(iter):# for(int it = 0; it<iteration; it++)
    # {	
		#print(f"Iteration: {it}")# printf("Iteration: %d \n:", it);

		# Read  depth(slice) plane d, update depth(slice) d+1
		for d in range(depth-1): # for(int d = 0; d < depth-1; d++)
		# {
			param_startInd = np.intc(1) #int startInd = 1;
			param_numInd   = np.intc(9) # int numInd = 9;
			param_updateSliceInd = np.intc(d+1) # int updateSliceInd = d+1;
			try:
				kernel_updateDistanceMap_Fwd_k(param_depth, param_height, param_width, 
					param_dd_dev, param_dh_dev, param_dw_dev, 
					param_im_stride_k, param_im_stride_j, param_im_stride_i, param_imgChannel_in, param_img_offset_dev,
					param_dm_stride_k, param_dm_stride_j, param_dm_stride_i, param_distChannel_in, param_dist_offset_dev, 
					param_lambda, param_local_dis_dev, param_startInd, param_numInd, param_updateSliceInd, param_tempDevMemory_img, param_tempDevMemory_dis,
					grid=a_GridDim_fixSlice, block=a_BlockDim_fixSlice)
			except Exception as ex:
				errorHappened = True
				print("kernel_updateDistanceMap_Fwd_k launch failed")
				logging.exception('Caught an error')
			finally:
				if(True == errorHappened):
					print("Returning with error")
					return distance_out
			try:
				cuda.Context.synchronize()
			except Exception as ex:
				errorHappened = True
				print("cudaDeviceSynchronize  failed")
				logging.exception('Caught an error')
			finally:
				if(True == errorHappened):
					print("Returning with error")
					return distance_out
		# }

		# Read  height(coronal) plane  h, update height(coronal) plane h+1
		for h in range(height-1): # for(int h = 0; h < height-1; h++)
		# {
			param_startInd = np.intc(19) #int startInd = 19;
			param_numInd   = np.intc(3) # int numInd = 3;
			param_updateCoronalInd = np.intc(h+1) # int updateCoronalInd = h+1;
			try:
				kernel_updateDistanceMap_Fwd_j(param_depth, param_height, param_width, 
					param_dd_dev, param_dh_dev, param_dw_dev, 
					param_im_stride_k, param_im_stride_j, param_im_stride_i, param_imgChannel_in, param_img_offset_dev,
					param_dm_stride_k, param_dm_stride_j, param_dm_stride_i, param_distChannel_in, param_dist_offset_dev, 
					param_lambda, param_local_dis_dev, param_startInd, param_numInd, param_updateCoronalInd, param_tempDevMemory_img, param_tempDevMemory_dis,
					grid=a_GridDim_fixCoronal, block=a_BlockDim_fixCoronal)
			except Exception as ex:
				errorHappened = True
				print("kernel_updateDistanceMap_Fwd_j launch failed")
				logging.exception('Caught an error')
			finally:
				if(True == errorHappened):
					print("Returning with error")
					return distance_out
			try:
				cuda.Context.synchronize()
			except Exception as ex:
				errorHappened = True
				print("cudaDeviceSynchronize  failed")
				logging.exception('Caught an error')
			finally:
				if(True == errorHappened):
					print("Returning with error")
					return distance_out
		# }

		# Read  width (sagittal) plane  w, update width (sagittal) plane w+1
		for w in range(width-1): # for(int w = 0; w < width-1; w++)
		# {
			param_startInd = np.intc(25) #int startInd = 25;
			param_numInd   = np.intc(1) # int numInd = 1;
			param_updateSagittalInd = np.intc(w+1) # int updateSagittalInd = w+1;
			try:
				kernel_updateDistanceMap_Fwd_i(param_depth, param_height, param_width, 
					param_dd_dev, param_dh_dev, param_dw_dev, 
					param_im_stride_k, param_im_stride_j, param_im_stride_i, param_imgChannel_in, param_img_offset_dev,
					param_dm_stride_k, param_dm_stride_j, param_dm_stride_i, param_distChannel_in, param_dist_offset_dev, 
					param_lambda, param_local_dis_dev, param_startInd, param_numInd, param_updateSagittalInd, param_tempDevMemory_img, param_tempDevMemory_dis,
					grid=a_GridDim_fixSagittal, block=a_BlockDim_fixSagittal)
			except Exception as ex:
				errorHappened = True
				print("kernel_updateDistanceMap_Fwd_i launch failed")
				logging.exception('Caught an error')
			finally:
				if(True == errorHappened):
					print("Returning with error")
					return distance_out
			try:
				cuda.Context.synchronize()
			except Exception as ex:
				errorHappened = True
				print("cudaDeviceSynchronize  failed")
				logging.exception('Caught an error')
			finally:
				if(True == errorHappened):
					print("Returning with error")
					return distance_out
		# }		

		# Read  depth(slice) plane d, update depth(slice) d-1
		for d in range(depth-1, 0, -1): # for(int d = depth-1; d > 0; d--)
		# {
			param_startInd = np.intc(10) #int startInd = 10;
			param_numInd   = np.intc(9) # int numInd = 9;
			param_updateSliceInd = np.intc(d-1) # int updateSliceInd = d-1;
			try:
				kernel_updateDistanceMap_Bwd_k(param_depth, param_height, param_width, 
					param_dd_dev, param_dh_dev, param_dw_dev, 
					param_im_stride_k, param_im_stride_j, param_im_stride_i, param_imgChannel_in, param_img_offset_dev,
					param_dm_stride_k, param_dm_stride_j, param_dm_stride_i, param_distChannel_in, param_dist_offset_dev, 
					param_lambda, param_local_dis_dev, param_startInd, param_numInd, param_updateSliceInd, param_tempDevMemory_img, param_tempDevMemory_dis,
					grid=a_GridDim_fixSlice, block=a_BlockDim_fixSlice)
			except Exception as ex:
				errorHappened = True
				print("kernel_updateDistanceMap_Bwd_k launch failed")
				logging.exception('Caught an error')
			finally:
				if(True == errorHappened):
					print("Returning with error")
					return distance_out
			try:
				cuda.Context.synchronize()
			except Exception as ex:
				errorHappened = True
				print("cudaDeviceSynchronize  failed")
				logging.exception('Caught an error')
			finally:
				if(True == errorHappened):
					print("Returning with error")
					return distance_out
		# }

		# Read  height(coronal) plane  h, update height(coronal) plane h-1
		for h in range(height-1, 0, -1): # for(int h = height-1; h > 0; h--)
		# {
			param_startInd = np.intc(22) #int startInd = 22;
			param_numInd   = np.intc(3) # int numInd = 3;
			param_updateCoronalInd = np.intc(h-1) # int updateCoronalInd = h-1;
			try:
				kernel_updateDistanceMap_Bwd_j(param_depth, param_height, param_width, 
					param_dd_dev, param_dh_dev, param_dw_dev, 
					param_im_stride_k, param_im_stride_j, param_im_stride_i, param_imgChannel_in, param_img_offset_dev,
					param_dm_stride_k, param_dm_stride_j, param_dm_stride_i, param_distChannel_in, param_dist_offset_dev, 
					param_lambda, param_local_dis_dev, param_startInd, param_numInd, param_updateCoronalInd, param_tempDevMemory_img, param_tempDevMemory_dis,
					grid=a_GridDim_fixCoronal, block=a_BlockDim_fixCoronal)
			except Exception as ex:
				errorHappened = True
				print("kernel_updateDistanceMap_Bwd_j launch failed")
				logging.exception('Caught an error')
			finally:
				if(True == errorHappened):
					print("Returning with error")
					return distance_out
			try:
				cuda.Context.synchronize()
			except Exception as ex:
				errorHappened = True
				print("cudaDeviceSynchronize  failed")
				logging.exception('Caught an error')
			finally:
				if(True == errorHappened):
					print("Returning with error")
					return distance_out
		# }

		# Read  width (sagittal) plane  w, update width (sagittal) plane w-1
		for w in range(width-1, 0, -1): # for(int w = width-1; w > 0; w--)
		# {
			param_startInd = np.intc(26) #int startInd = 26;
			param_numInd   = np.intc(1) # int numInd = 1;
			param_updateSagittalInd = np.intc(w-1) # int updateSagittalInd = w-1;
			try:
				kernel_updateDistanceMap_Bwd_i(param_depth, param_height, param_width, 
					param_dd_dev, param_dh_dev, param_dw_dev, 
					param_im_stride_k, param_im_stride_j, param_im_stride_i, param_imgChannel_in, param_img_offset_dev,
					param_dm_stride_k, param_dm_stride_j, param_dm_stride_i, param_distChannel_in, param_dist_offset_dev, 
					param_lambda, param_local_dis_dev, param_startInd, param_numInd, param_updateSagittalInd, param_tempDevMemory_img, param_tempDevMemory_dis,
					grid=a_GridDim_fixSagittal, block=a_BlockDim_fixSagittal)
			except Exception as ex:
				errorHappened = True
				print("kernel_updateDistanceMap_Bwd_i launch failed")
				logging.exception('Caught an error')
			finally:
				if(True == errorHappened):
					print("Returning with error")
					return distance_out
			try:
				cuda.Context.synchronize()
			except Exception as ex:
				errorHappened = True
				print("cudaDeviceSynchronize  failed")
				logging.exception('Caught an error')
			finally:
				if(True == errorHappened):
					print("Returning with error")
					return distance_out
		# }
	# }

	# Transfer modified distance  from device  to host distance pointer	
	if(False == errorHappened):
		cuda.memcpy_dtoh(distance, param_tempDevMemory_dis)
		#Result between GeodisTk (CPU Geodesic) and this GPU version showing difference associated with axis switch if I leave like this
		#So attempting axis match
		distance_out = np.transpose(distance, (2,1,0)).copy()
		print("GpuGeodisKernel is successful")
		# #For Testing
		# for click in clickLocs:
		# 	print(f"seed value and distance at click loc {click} : seed: {seeds[click]}  dis: {distance[click]}")
		# for click in unClickLocs:
		# 	print(f"seed value and distance at UN-click loc {click} : seed: {seeds[click]}  dis: {distance[click]}")
	else:
		print("GpuGeodisKernel failed")

	# int depth, int height, int width, int x, int y, int z,
	# int *dd_dev, int *dh_dev, int *dw_dev, 
    # long im_stride_k,  long im_stride_j,  long im_stride_i, int imgChannel_in, long *img_offset_dev,
	# long dm_stride_k, long dm_stride_j, long dm_stride_i, int distChannel_in, long *dist_offset_dev,   
	# float lambda, float *local_dis_dev, int startInd, int numInd,  float *tempDevMemory_img,  float *tempDevMemory_dis

	#Clean-up memory
	param_dd_dev.free()# if(NULL != dd_dev){ cudaFree(dd_dev); dd_dev = NULL;}
	param_dh_dev.free()# if(NULL != dh_dev){ cudaFree(dh_dev); dh_dev = NULL;}
	param_dw_dev.free()# if(NULL != dw_dev){ cudaFree(dw_dev); dw_dev = NULL;}
	param_local_dis_dev.free()# if(NULL != local_dis_dev){ cudaFree(local_dis_dev); local_dis_dev = NULL;}
	param_img_offset_dev.free()# if(NULL != img_offset_dev){ cudaFree(img_offset_dev); img_offset_dev = NULL;}
	param_dist_offset_dev.free()# if(NULL != dist_offset_dev){ cudaFree(dist_offset_dev); dist_offset_dev = NULL;}	
	param_tempDevMemory_dis.free()# if(NULL != tempDevMemory_dis){ cudaFree(tempDevMemory_dis); tempDevMemory_dis = NULL;}
	param_tempDevMemory_seed.free()# if(NULL != tempDevMemory_seed){ cudaFree(tempDevMemory_seed); tempDevMemory_seed = NULL;}
	param_tempDevMemory_img.free()# if(NULL != tempDevMemory_img){ cudaFree(tempDevMemory_img); tempDevMemory_img = NULL;}

	return distance_out



######################### Original C++ Kernel wrapper ###############################
# /**
#  * Wrapper function for the CUDA kernel function.
#  * @param img multichannel float valued input CT-PET volume: img.
#  * @param seeds single channel unsigned char valued input binary mask volume: seeds).
#  * @param distance single channel  float valued output distance map volume: dist.
#  * @param depth int valued depth of the volume(s)
#  * @param height int valued height of the volume(s)
#  * @param width int valued width of the volume(s)
#  * @param channel int valued number of channels  of the img volume
#  * @param spacing float  vector (size=3) of spacing along depth, height, width direction
#  * @param lambda float valued mix (between 0.0 and 1.0) of Euclidean and Geodesic distance: 0 => full Euclidean, 1 => full Geodesic
#  * @param iter int valued iteration number
#  **/
# void GpuGeodisKernel(const float * img, const unsigned char * seeds, float * distance,
#                             int depth, int height, int width, int channel, 
#                             std::vector<float> spacing, float lambda, int iteration) 
# {
# 	cudaError_t cudaStatus = cudaSuccess;	
# 	int aCase = 1;	
# 	// Compute various pitches / strides to navigate single and multichannel data (seed, image, distance map)
# 	// Acronyms: dm : distance map; im : image; sm: seed map; stride : basePitch; 
# 	// byte size per voxel per channel for image and distance = 4, since float valued voxel per channel 
# 	// C (num channel in input image)
#   // stride_k = H*W; stride_j = W; stride_i = 1; 
# 	// sm_stride_k = H*W;   sm_stride_j = W;   sm_stride_i = 1;  Seedmap      is single channel, uChar valued
# 	// im_stride_k = H*W*C; im_stride_j = W*C; im_stride_i = C;  Image        is multi  channel, float valued
# 	// dm_stride_k = H*W;   dm_stride_j = W;   dm_stride_i = 1;  Distance map is single channel  float valued	
# 	int imgChannel_in = channel;
# 	int distChannel_in = 1;
# 	long stride_k = height*width; long stride_j = width; long  stride_i = 1;
# 	long sm_stride_k = stride_k;                long sm_stride_j = stride_j;                long sm_stride_i = stride_i;
# 	long im_stride_k = stride_k*imgChannel_in;  long im_stride_j = stride_j*imgChannel_in;  long im_stride_i = stride_i*imgChannel_in;
# 	long dm_stride_k = stride_k*distChannel_in; long dm_stride_j = stride_j*distChannel_in; long dm_stride_i = stride_i*distChannel_in;
	
# 	// float sqrt3 = sqrt(3.0);
# 	// float sqrt2 = sqrt(2.0);
#   // float sqrt1 = 1.0;	
# 	float s0Sqr = spacing[0] *spacing[0];
#   float s1Sqr = spacing[1] *spacing[1];
#   float s2Sqr = spacing[2] *spacing[2];
    
# 	//Local distances for 26-neighbourhood of a voxel are now computed and ARRANGED according the way the update kernels
# 	//will access them
# 	// Accessed by         <--Center--> <-----Update  k-forward-----------> <-----Update k-backward------------>  <--Ud j-Dwn --> <--Ud j-Up--> <-- Ud i-R --> <-- Ud i-L -->  
# 	//             INDEX    0            1   2   3   4   5   6   7   8   9    10  11  12  13  14  15  16  17  18    19   20   21   22   23  24        25           26 
# 	// Read from depth(k)   0,          -1, -1, -1, -1, -1, -1, -1, -1, -1,   +1, +1, +1, +1, +1, +1, +1, +1, +1,    0,   0,   0,   0,   0,  0,        0,           0
# 	// Read from height(j)  0,          -1, -1, -1,  0,  0,  0, +1, +1, +1,   -1, -1, -1,  0,  0,  0, +1, +1, +1,   -1,  -1,  -1,  +1,  +1, +1,        0,           0
# 	// Read from width(i)   0,          -1,  0, +1, -1,  0, +1, -1,  0, +1,   -1,  0, +1, -1,  0, +1, -1,  0, +1,   -1,   0,  +1,  -1,   0, +1,       -1,          +1
# 	// int lookupId_Fwd_k[9] = {1,2,3,4,5,6,7,8,9};
# 	// int lookupId_Bwd_k[9] = {10,11,12,13,14,15,16,17,18};
# 	// int lookupId_Fwd_j[3] = {19,20,21};
# 	// int lookupId_Bwd_j[3] = {22,23,24};
# 	// int lookupId_Fwd_i[1] = {25};
# 	// int lookupId_Bwd_i[1] = {26};
# 	int dd_[27] = {0, -1, -1, -1, -1, -1, -1, -1, -1, -1, +1, +1, +1, +1, +1, +1, +1, +1, +1,  0,  0,  0,  0,  0,  0,  0,  0};
# 	int dh_[27] = {0, -1, -1, -1,  0,  0,  0, +1, +1, +1, -1, -1, -1,  0,  0,  0, +1, +1, +1, -1, -1, -1, +1, +1, +1,  0,  0};
# 	int dw_[27] = {0, -1,  0, +1, -1,  0, +1, -1,  0, +1, -1,  0, +1, -1,  0, +1, -1,  0, +1, -1,  0, +1, -1,  0, +1, -1, +1};
# 	long dist_offset_[27];
# 	for(int i = 0; i< 27; i++){
#         dist_offset_[i] = dd_[i] * dm_stride_k + dh_[i] * dm_stride_j + dw_[i] * dm_stride_i;
#     }
# 	long img_offset_[27];
# 	for(int i = 0; i< 27; i++){
#         img_offset_[i] = dd_[i] * im_stride_k + dh_[i] * im_stride_j  + dw_[i] * im_stride_i;
#     }
# 	float local_dis_[27];
# 	for(int i = 0; i< 27; i++){
#         float distance = 0.0;
#         if(dd_[i] !=0) distance += s0Sqr; //spacing[0] *spacing[0];
#         if(dh_[i] !=0) distance += s1Sqr; //spacing[1] *spacing[1];
#         if(dw_[i] !=0) distance += s2Sqr; //spacing[2] *spacing[2];
#         distance = sqrt(distance);
#         local_dis_[i] = distance;
#     }	

# 	//Assign threads per blocks and GridDim  for launching kernel
# 	dim3 a_BlockDim_vol = dim3(8, 8, 8);
# 	dim3 a_GridDim_vol = dim3(width / a_BlockDim_vol.x, height / a_BlockDim_vol.y, depth / a_BlockDim_vol.z);

# 	dim3 a_BlockDim_fixSlice = dim3(8, 8, 1);
# 	dim3 a_GridDim_fixSlice = dim3(width / a_BlockDim_fixSlice.x, height / a_BlockDim_fixSlice.y, 1);
# 	dim3 a_BlockDim_fixCoronal = dim3(8,1,8);
# 	dim3 a_GridDim_fixCoronal = dim3(width / a_BlockDim_fixCoronal.x, 1, depth / a_BlockDim_fixCoronal.z);
# 	dim3 a_BlockDim_fixSagittal = dim3(1, 8, 8);
# 	dim3 a_GridDim_fixSagittal = dim3(1, height / a_BlockDim_fixSagittal.y, depth / a_BlockDim_fixSagittal.z);

# 	int *dd_dev = NULL;
# 	int *dh_dev = NULL;
# 	int *dw_dev = NULL;
# 	long *dist_offset_dev = NULL;
# 	long *img_offset_dev = NULL;
# 	float *local_dis_dev = NULL;
# 	unsigned char *tempDevMemory_seed = NULL;
# 	float *tempDevMemory_img = NULL;
# 	float *tempDevMemory_dis = NULL;

# 	switch(aCase)
# 	{
# 		case 1:
# 		{
# 			//Create device memory to hold dd_ and transfer dd_ from host to device
# 			cudaStatus = cudaMalloc((void**)&dd_dev, sizeof(int)*27);
# 			if ( cudaSuccess != cudaStatus) { fprintf(stderr, "Error in creating dd_dev. Returning false."); break;}
# 			else{cudaStatus = cudaMemcpy((void*)dd_dev, (const void*)dd_, sizeof(int)*27, cudaMemcpyHostToDevice);
# 				if ( cudaSuccess != cudaStatus) {fprintf(stderr, "Error in cudaMemcpy of dd_ from host to device failed. Returning false."); break;}
# 			}
# 			//Create device memory to hold dh_ and transfer dh_ from host to device
# 			cudaStatus = cudaMalloc((void**)&dh_dev, sizeof(int)*27);
# 			if ( cudaSuccess != cudaStatus) { fprintf(stderr, "Error in creating dh_dev. Returning false."); break;}
# 			else{cudaStatus = cudaMemcpy((void*)dh_dev, (const void*)dh_, sizeof(int)*27, cudaMemcpyHostToDevice);
# 				if ( cudaSuccess != cudaStatus) {fprintf(stderr, "Error in cudaMemcpy of dh_ from host to device failed. Returning false."); break;}
# 			}
# 			//Create device memory to hold dw_ and transfer dw_ from host to device
# 			cudaStatus = cudaMalloc((void**)&dw_dev, sizeof(int)*27);
# 			if ( cudaSuccess != cudaStatus) { fprintf(stderr, "Error in creating dw_dev. Returning false."); break;}
# 			else{cudaStatus = cudaMemcpy((void*)dw_dev, (const void*)dw_, sizeof(int)*27, cudaMemcpyHostToDevice);
# 				if ( cudaSuccess != cudaStatus) {fprintf(stderr, "Error in cudaMemcpy of dw_ from host to device failed. Returning false."); break;}
# 			}
# 			//Create device memory to hold dist_offset and transfer dist_offset from host to device
# 			cudaStatus = cudaMalloc((void**)&dist_offset_dev, sizeof(long)*27);
# 			if ( cudaSuccess != cudaStatus) { fprintf(stderr, "Error in creating dist_offset_dev. Returning false."); break;}
# 			else{cudaStatus = cudaMemcpy((void*)dist_offset_dev, (const void*)dist_offset_, sizeof(long)*27, cudaMemcpyHostToDevice);
# 				if ( cudaSuccess != cudaStatus) {fprintf(stderr, "Error in cudaMemcpy of dist_offset from host to device failed. Returning false."); break;}
# 			}
# 			//Create device memory to hold img_offset_ and transfer img_offset_ from host to device
# 			cudaStatus = cudaMalloc((void**)&img_offset_dev, sizeof(long)*27);
# 			if ( cudaSuccess != cudaStatus) { fprintf(stderr, "Error in creating img_offset_dev. Returning false."); break;}
# 			else{cudaStatus = cudaMemcpy((void*)img_offset_dev, (const void*)img_offset_, sizeof(long)*27, cudaMemcpyHostToDevice);
# 				if ( cudaSuccess != cudaStatus) {fprintf(stderr, "Error in cudaMemcpy of img_offset_ from host to device failed. Returning false."); break;}
# 			}
# 			//Create device memory to hold local_dis_ and transfer local_dis_ from host to device
# 			cudaStatus = cudaMalloc((void**)&local_dis_dev, sizeof(float)*27);
# 			if ( cudaSuccess != cudaStatus) { fprintf(stderr, "Error in creating local_dis_dev. Returning false."); break;}
# 			else{cudaStatus = cudaMemcpy((void*)local_dis_dev, (const void*)local_dis_, sizeof(float)*27, cudaMemcpyHostToDevice);
# 				if ( cudaSuccess != cudaStatus) {fprintf(stderr, "Error in cudaMemcpy of image from host to device failed. Returning false."); break;}
# 			}
# 			//Create device memory to hold seeds and transfer seeds from host to device
# 			cudaStatus = cudaMalloc((void**)&tempDevMemory_seed, sizeof(unsigned char)*width*height*depth);
# 			if ( cudaSuccess != cudaStatus){fprintf(stderr, "Error in creating tempDevMemory_seed. Returning false.");break;}
# 			else {cudaStatus = cudaMemcpy((void*)tempDevMemory_seed, (const void*)seeds, sizeof(unsigned char)*width*height*depth, cudaMemcpyHostToDevice);
# 				if ( cudaSuccess != cudaStatus){fprintf(stderr, "Error in cudaMemcpy of seeds from host to device failed. Returning false.");break;}
# 			}	
# 			//Create device memory to hold img and transfer img from host to device
# 			cudaStatus = cudaMalloc((void**)&tempDevMemory_img, sizeof(float)*imgChannel_in*width*height*depth);
# 			if ( cudaSuccess != cudaStatus) {fprintf(stderr, "Error in creating tempDevMemory_img. Returning false.");break;}
# 			else{cudaStatus = cudaMemcpy((void*)tempDevMemory_img, (const void*)img, sizeof(float)*imgChannel_in*width*height*depth, cudaMemcpyHostToDevice);
# 				if ( cudaSuccess != cudaStatus){fprintf(stderr, "Error in cudaMemcpy of img from host to device failed. Returning false."); break;}
# 			}				
# 			//Create device memory to hold distance
# 			cudaStatus = cudaMalloc((void**)&tempDevMemory_dis, sizeof(float)*distChannel_in*width*height*depth);
# 			if ( cudaSuccess != cudaStatus) {fprintf(stderr, "Error in creating tempDevMemory_dis. Returning false.");break;}

# 			printf("Initialize distance map using seed location.\n:");
# 			//launch kernel to initialize to distance map based on seed.
# 			kernel_setInitialDistanceBasedOnSeed<<<a_GridDim_vol, a_BlockDim_vol>>>(
# 					depth,
# 					height,
# 					width,
# 					sm_stride_j, //long ucharsPerRow_seed,
# 					sm_stride_k, //long ucharsPerSlice_seed,
# 					tempDevMemory_seed, //unsigned char *devSeed,
# 					distChannel_in,
# 					dm_stride_j, //floatsPerRow_dis,
# 					dm_stride_k, //floatsPerSlice_dis,					
# 					tempDevMemory_dis);
# 			cudaStatus = cudaGetLastError();
# 			if ( cudaSuccess != cudaStatus){fprintf(stderr, "kernel_setInitialDistanceBasedOnSeed launch failed: %s\n", cudaGetErrorString(cudaStatus));break;}

# 			for(int it = 0; it<iteration; it++)
#     		{
# 				printf("Iteration: %d \n:", it);
# 				//Read  depth(slice) plane d, update depth(slice) d+1
# 				for(int d = 0; d < depth-1; d++)
#         		{
# 					int startInd = 1;
# 					int numInd = 9;
# 					int updateSliceInd = d+1;
# 					kernel_updateDistanceMap_Fwd_k<<<a_GridDim_fixSlice,a_BlockDim_fixSlice>>>(depth, height, width, 
# 						dd_dev, dh_dev, dw_dev, 
# 						im_stride_k, im_stride_j, im_stride_i, imgChannel_in, img_offset_dev,
# 						dm_stride_k, dm_stride_j, dm_stride_i, distChannel_in, dist_offset_dev, 
# 						lambda, local_dis_dev, startInd, numInd, updateSliceInd, tempDevMemory_img, tempDevMemory_dis);
# 					// Check for any errors launching the kernel
# 					cudaStatus = cudaGetLastError();
# 					if ( cudaSuccess != cudaStatus) {fprintf(stderr, "kernel_updateDistanceMap_Fwd_k launch failed: %s\n", cudaGetErrorString(cudaStatus));break;}    
# 					// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
# 					cudaStatus = cudaDeviceSynchronize();
# 					if ( cudaSuccess != cudaStatus) {fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel_updateDistanceMap_Fwd_k!\n", cudaStatus);break;}
# 				}
				
# 				//Read  height(coronal) plane  h, update height(coronal) plane h+1
# 				for(int h = 0; h < height-1; h++)
#         		{
# 					int startInd = 19;
# 					int numInd = 3;
# 					int updateCoronalInd = h+1;
# 					kernel_updateDistanceMap_Fwd_j<<<a_GridDim_fixCoronal,a_BlockDim_fixCoronal>>>(depth, height, width, 
# 						dd_dev, dh_dev, dw_dev, 
# 						im_stride_k, im_stride_j, im_stride_i, imgChannel_in, img_offset_dev,
# 						dm_stride_k, dm_stride_j, dm_stride_i, distChannel_in, dist_offset_dev, 
# 						lambda, local_dis_dev, startInd, numInd, updateCoronalInd, tempDevMemory_img, tempDevMemory_dis);
# 					// Check for any errors launching the kernel
# 					cudaStatus = cudaGetLastError();
# 					if ( cudaSuccess != cudaStatus) {fprintf(stderr, "kernel_updateDistanceMap_Fwd_j launch failed: %s\n", cudaGetErrorString(cudaStatus));break;}    
# 					// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
# 					cudaStatus = cudaDeviceSynchronize();
# 					if ( cudaSuccess != cudaStatus) {fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel_updateDistanceMap_Fwd_j!\n", cudaStatus);break;}
# 				}				
				
# 				//Read  width (sagittal) plane  w, update width (sagittal) plane w+1
# 				for(int w = 0; w < width-1; w++)
#         		{
# 					int startInd = 25;
# 					int numInd = 1;
# 					int updateSagittalInd = w+1;
# 					kernel_updateDistanceMap_Fwd_i<<<a_GridDim_fixSagittal,a_BlockDim_fixSagittal>>>(depth, height, width, 
# 						dd_dev, dh_dev, dw_dev, 
# 						im_stride_k, im_stride_j, im_stride_i, imgChannel_in, img_offset_dev,
# 						dm_stride_k, dm_stride_j, dm_stride_i, distChannel_in, dist_offset_dev, 
# 						lambda, local_dis_dev, startInd, numInd, updateSagittalInd, tempDevMemory_img, tempDevMemory_dis);
# 					// Check for any errors launching the kernel
# 					cudaStatus = cudaGetLastError();
# 					if ( cudaSuccess != cudaStatus) {fprintf(stderr, "kernel_updateDistanceMap_Fwd_i launch failed: %s\n", cudaGetErrorString(cudaStatus));break;}    
# 					// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
# 					cudaStatus = cudaDeviceSynchronize();
# 					if ( cudaSuccess != cudaStatus) {fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel_updateDistanceMap_Fwd_i!\n", cudaStatus);break;}
# 				}
				
# 				//Read  depth(slice) plane d, update depth(slice) d-1
# 				for(int d = depth-1; d > 0; d--)
#         		{
# 					int startInd = 10;
# 					int numInd = 9;
# 					int updateSliceInd = d-1;
# 					kernel_updateDistanceMap_Bwd_k<<<a_GridDim_fixSlice,a_BlockDim_fixSlice>>>(depth, height, width, 
# 						dd_dev, dh_dev, dw_dev, 
# 						im_stride_k, im_stride_j, im_stride_i, imgChannel_in, img_offset_dev,
# 						dm_stride_k, dm_stride_j, dm_stride_i, distChannel_in, dist_offset_dev, 
# 						lambda, local_dis_dev, startInd, numInd, updateSliceInd, tempDevMemory_img, tempDevMemory_dis);
# 					// Check for any errors launching the kernel
# 					cudaStatus = cudaGetLastError();
# 					if ( cudaSuccess != cudaStatus) {fprintf(stderr, "kernel_updateDistanceMap_Bwd_k launch failed: %s\n", cudaGetErrorString(cudaStatus));break;}    
# 					// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
# 					cudaStatus = cudaDeviceSynchronize();
# 					if ( cudaSuccess != cudaStatus) {fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel_updateDistanceMap_Bwd_k!\n", cudaStatus);break;}
# 				}

# 				//Read  height(coronal) plane  h, update height(coronal) plane h-1
# 				for(int h = height-1; h > 0; h--)
#         		{
# 					int startInd = 22;
# 					int numInd = 3;
# 					int updateCoronalInd = h-1;
# 					kernel_updateDistanceMap_Bwd_j<<<a_GridDim_fixCoronal,a_BlockDim_fixCoronal>>>(depth, height, width, 
# 						dd_dev, dh_dev, dw_dev, 
# 						im_stride_k, im_stride_j, im_stride_i, imgChannel_in, img_offset_dev,
# 						dm_stride_k, dm_stride_j, dm_stride_i, distChannel_in, dist_offset_dev, 
# 						lambda, local_dis_dev, startInd, numInd, updateCoronalInd, tempDevMemory_img, tempDevMemory_dis);
# 					// Check for any errors launching the kernel
# 					cudaStatus = cudaGetLastError();
# 					if ( cudaSuccess != cudaStatus) {fprintf(stderr, "kernel_updateDistanceMap_Bwd_j launch failed: %s\n", cudaGetErrorString(cudaStatus));break;}    
# 					// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
# 					cudaStatus = cudaDeviceSynchronize();
# 					if ( cudaSuccess != cudaStatus) {fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel_updateDistanceMap_Bwd_j!\n", cudaStatus);break;}
# 				}

# 				//Read  width (sagittal) plane  w, update width (sagittal) plane w-1
# 				for(int w = width-1; w > 0; w--)
#         		{
# 					int startInd = 26;
# 					int numInd = 1;
# 					int updateSagittalInd = w-1;
# 					kernel_updateDistanceMap_Bwd_i<<<a_GridDim_fixSagittal,a_BlockDim_fixSagittal>>>(depth, height, width, 
# 						dd_dev, dh_dev, dw_dev, 
# 						im_stride_k, im_stride_j, im_stride_i, imgChannel_in, img_offset_dev,
# 						dm_stride_k, dm_stride_j, dm_stride_i, distChannel_in, dist_offset_dev, 
# 						lambda, local_dis_dev, startInd, numInd, updateSagittalInd, tempDevMemory_img, tempDevMemory_dis);
# 					// Check for any errors launching the kernel
# 					cudaStatus = cudaGetLastError();
# 					if ( cudaSuccess != cudaStatus) {fprintf(stderr, "kernel_updateDistanceMap_Bwd_i launch failed: %s\n", cudaGetErrorString(cudaStatus));break;}    
# 					// cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
# 					cudaStatus = cudaDeviceSynchronize();
# 					if ( cudaSuccess != cudaStatus) {fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernel_updateDistanceMap_Bwd_i!\n", cudaStatus);break;}
# 				}
# 			}
# 			//Transfer modified distance  from device  to host distance pointer
# 			cudaStatus = cudaMemcpy((void*)distance, (const void*)tempDevMemory_dis, sizeof(float)*distChannel_in*width*height*depth, cudaMemcpyDeviceToHost);
# 			if ( cudaSuccess != cudaStatus) {
# 				fprintf(stderr, "cudaMemcpy from device to host  failed: %s\n", cudaGetErrorString(cudaStatus));
# 				break;
# 			}
# 		}//End case
# 	}//End switch
# 	if(NULL != dd_dev){ cudaFree(dd_dev); dd_dev = NULL;}
# 	if(NULL != dh_dev){ cudaFree(dh_dev); dh_dev = NULL;}
# 	if(NULL != dw_dev){ cudaFree(dw_dev); dw_dev = NULL;}
# 	if(NULL != tempDevMemory_dis){ cudaFree(tempDevMemory_dis); tempDevMemory_dis = NULL;}
# 	if(NULL != tempDevMemory_seed){ cudaFree(tempDevMemory_seed); tempDevMemory_seed = NULL;}
# 	if(NULL != tempDevMemory_img){ cudaFree(tempDevMemory_img); tempDevMemory_img = NULL;}
# 	if(NULL != local_dis_dev){ cudaFree(local_dis_dev); local_dis_dev = NULL;}
# 	if(NULL != img_offset_dev){ cudaFree(img_offset_dev); img_offset_dev = NULL;}
# 	if(NULL != dist_offset_dev){ cudaFree(dist_offset_dev); dist_offset_dev = NULL;}
# 	if(cudaSuccess == cudaStatus)
# 		fprintf(stderr, "GpuGeodisKernel is successful\n");
# 	else
# 		fprintf(stderr, "GpuGeodisKernel failed\n");
# 	return;
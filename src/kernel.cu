#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <algorithm>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAError() \
  do \
  { \
    if(cudaPeekAtLastError() != cudaSuccess)  \
    { \
      std::cerr << cudaGetErrorString(cudaPeekAtLastError()) << __FILE__ << __LINE__ << "\n"; \
      exit(-1); \
    } \
  } while(0) \

/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

//number of neighboring grid cells to check
//LOOK-2.1
//having to check 27 grid cells would require more checks in the loop, so it
//would be less efficient than checking 8 grid cells
#define CHECK_8 0
#ifdef CHECK_8
constexpr int NEIGHBORS_TO_CHECK = 8;
constexpr int NEIGHBORS_TO_CHECK_WIDTH = 2;
#else
constexpr int NEIGHBORS_TO_CHECK = 27;
constexpr int NEIGHBORS_TO_CHECK_WIDTH = 3;
#endif

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

//pos and vel sorted by array indices
glm::vec3* dev_vel_sorted;
glm::vec3* dev_pos_sorted;

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash(static_cast<int>(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3(static_cast<float>(unitDistrib(rng)), static_cast<float>(unitDistrib(rng)), static_cast<float>(unitDistrib(rng)));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  const int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    const glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc(reinterpret_cast<void**>(&dev_pos), N * sizeof(glm::vec3));
  checkCUDAError();

  cudaMalloc(reinterpret_cast<void**>(&dev_vel1), N * sizeof(glm::vec3));
  checkCUDAError();

  cudaMalloc(reinterpret_cast<void**>(&dev_vel2), N * sizeof(glm::vec3));
  checkCUDAError();

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAError();

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = static_cast<int>(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;
  std::cout << gridCellCount << "-" << gridCellWidth << "-" << gridMinimum.x << "-" << gridMinimum.y << "-" << gridMinimum.z << "\n";
  //10648-10--110--110--110
  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.

  cudaMalloc(reinterpret_cast<void**>(&dev_particleArrayIndices), N * sizeof(int));
  checkCUDAError();

  cudaMalloc(reinterpret_cast<void**>(&dev_particleGridIndices), N * sizeof(int));
  checkCUDAError();

  cudaMalloc(reinterpret_cast<void**>(&dev_gridCellStartIndices), gridCellCount * sizeof(int));
  checkCUDAError();

  cudaMalloc(reinterpret_cast<void**>(&dev_gridCellEndIndices), gridCellCount * sizeof(int));
  checkCUDAError();

  //part 2.3 allocate memory for position and velocity struct
  cudaMalloc(reinterpret_cast<void**>(&dev_vel_sorted), N * sizeof(*dev_vel_sorted));
  checkCUDAError();

  cudaMalloc(reinterpret_cast<void**>(&dev_pos_sorted), N * sizeof(*dev_pos_sorted));
  checkCUDAError();

  cudaDeviceSynchronize();
  checkCUDAError();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAError();

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

//swap pointers (helps with ping pong)
template<typename T, typename std::enable_if<std::is_pointer<T>::value>::type* = nullptr>
void swap_pointers(T& p1, T& p2)
{
  T temp = p1;
  p1 = p2;
  p2 = temp;
}

//clamps the vec3 by normalizing it
template<typename T>
__device__ glm::vec3 clamp_vec3(T&& vec)
{
  if(glm::length(vec) > maxSpeed)
  {
    return glm::normalize(std::forward<T>(vec)) * maxSpeed;
  }
  return vec;
}

//helper function to check rule
template<typename CheckSuccessCallback>
__device__ void check_rule(float rule_distance, int this_boid, int other_boid, const glm::vec3* pos, CheckSuccessCallback check_success_callback)
{
  const auto& this_boid_pos = pos[this_boid];
  const auto& other_boid_pos = pos[other_boid];

  if (this_boid != other_boid && glm::distance(this_boid_pos, other_boid_pos) < rule_distance)
  {
    check_success_callback();
  }
}

//The following 3 functions checks two
//boids against each rule
__device__ void check_rule1(int this_boid, int other_boid, const glm::vec3* pos, glm::vec3& perceived_center, int& num_neighbors)
{
  check_rule(rule1Distance, this_boid, other_boid, pos,
             [&]()
             {
               const auto& other_boid_pos = pos[other_boid];
               perceived_center += other_boid_pos;
               num_neighbors++;
             });
}

__device__ void check_rule2(int this_boid, int other_boid, const glm::vec3* pos, glm::vec3& c)
{
  check_rule(rule2Distance, this_boid, other_boid, pos,
             [&]()
             {
               const auto& this_boid_pos = pos[this_boid];
               const auto& other_boid_pos = pos[other_boid];
               c -= (other_boid_pos - this_boid_pos);
             });
}

__device__ void check_rule3(int this_boid, int other_boid, const glm::vec3* pos, glm::vec3& perceived_velocity, int& num_neighbors,
                            const glm::vec3* vel)
{
  check_rule(rule3Distance, this_boid, other_boid, pos,
             [&]()
             {
               perceived_velocity += vel[other_boid];
               num_neighbors++;
             });
}

//The following 3 functions computes the
//rule velocity after all the boids in the
//area have been iterated through
__device__ glm::vec3 finish_rule1(const glm::vec3& this_boid_pos, glm::vec3& perceived_center, int& num_neighbors)
{
  if(num_neighbors)
  {
    perceived_center /= num_neighbors;
    return (perceived_center - this_boid_pos) * rule1Scale;
  }

  return {};
}

__device__ glm::vec3 finish_rule2(const glm::vec3& c)
{
  return c * rule2Scale;
}

__device__ glm::vec3 finish_rule3(glm::vec3& perceived_velocity, int& num_neighbors)
{
  if (num_neighbors)
  {
    perceived_velocity /= num_neighbors;
    return perceived_velocity * rule3Scale;
  }

  return {};
}

//The following 3 functions computes each rule naively (iterate through all boids)
__device__ glm::vec3 compute_rule1_naive(int N, int this_boid, const glm::vec3 *pos, const glm::vec3 *vel)
{
  glm::vec3 perceived_center{};

  auto num_neighbors = 0;

  for (auto other_boid = 0; other_boid < N; other_boid++)
  {
    check_rule1(this_boid, other_boid, pos, perceived_center, num_neighbors);
  }

  const auto& this_boid_pos = pos[this_boid];

  return finish_rule1(this_boid_pos, perceived_center, num_neighbors);
}

__device__ glm::vec3 compute_rule2_naive(int N, int this_boid, const glm::vec3 *pos, const glm::vec3 *vel)
{
  glm::vec3 c{};

  for (auto other_boid = 0; other_boid < N; other_boid++)
  {
    check_rule2(this_boid, other_boid, pos, c);

  }

  return finish_rule2(c);
}

__device__ glm::vec3 compute_rule3_naive(int N, int this_boid, const glm::vec3 *pos, const glm::vec3 *vel)
{
  glm::vec3 perceived_velocity{};

  auto num_neighbors = 0;

  for (auto other_boid = 0; other_boid < N; other_boid++)
  {
    check_rule3(this_boid, other_boid, pos, perceived_velocity, num_neighbors, vel);
  }

  return finish_rule3(perceived_velocity, num_neighbors);
}

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/

__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  // Rule 2: boids try to stay a distance d away from each other
  // Rule 3: boids try to match the speed of surrounding boids

  //add the result of all the rules
  auto result = compute_rule1_naive(N, iSelf, pos, vel) + 
    compute_rule2_naive(N, iSelf, pos, vel) + 
    compute_rule3_naive(N, iSelf, pos, vel);
  return result;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  // Compute a new velocity based on pos and vel1
  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1?

  // If vel1 is updated, then another GPU thread can update the vel1
  // and that would cause inconsistency.

  const int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  const auto vel = vel1[index] + computeVelocityChange(N, index, pos, vel1);
  vel2[index] = clamp_vec3(vel);
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  const int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
//probably
//for(z)
//  for(y)
//    for(x)
//because x is constantly changing, while y and z
//are less frequently changing in the loop
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
  
  const int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  
  //set the indices of the boid to match the thread index (same index as the position and velocity indices)
  indices[index] = index;

  //pos[index] - gridMin is the boid's position, with instead of
  //offsetting at [-100:100], it's offseted at [0:200]
  const auto b_pos = pos[index] - gridMin;

  //get boid's position index or "grid cube" that the boid is in (truncated to an int)
  const glm::ivec3 b_pos_index = b_pos * inverseCellWidth;

  //the position is converted to a 1d int (more efficient instead of holding the entire vec3)   
  gridIndices[index] = gridIndex3Dto1D(b_pos_index.x, b_pos_index.y, b_pos_index.z, gridResolution);
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

//how to expand variadic templates in c++11
//https://stackoverflow.com/questions/41623422/c-expand-variadic-template-arguments-into-a-statement
template<typename Type = int, typename... Ts>
__device__ auto truncate_to(Ts*... ts)
{
  (void)std::initializer_list<int>{(*ts = static_cast<Type>(*ts), 0)...};
}

//truncate glm vec to int
template<typename Vec, typename ConvertToType = int>
__device__ void truncate_glm_vec(Vec& vec)
{
  truncate_to<ConvertToType>(&vec.x, &vec.y, &vec.z);
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"

  const int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  //get the sorted boid grid value
  const auto previous_grid_cell_value = particleGridIndices[index - 1];
  const auto grid_cell_value = particleGridIndices[index];

  //if we are at the first grid index, 
  //we know that the grid index must be in a start index
  if(!index)
  {
    gridCellStartIndices[grid_cell_value] = index;
  }
  //get the previous element, and check if they belong to the same grid
  //if not, then we know that the grid_cell_value must be the 
  //start of a new grid index while previous_grid_cell_value must be
  //the last element to the previous grid cell group
  //c++17 if initializers: (VS15 doesn't have that support =/)
  //if(const auto previous_grid_cell = ...; previous_grid_cell != grid_cell)
  else
  {
    if (previous_grid_cell_value != grid_cell_value)
    {
      gridCellStartIndices[grid_cell_value] = index;
      gridCellEndIndices[previous_grid_cell_value] = index;
    }
    
    //still need to check for the last grid element and set
    //the end indices to point to this grid index
    else if(index == N - 1)
    {
      gridCellEndIndices[grid_cell_value] = index;
    }
  }
}

template<int neighbor_max_size = NEIGHBORS_TO_CHECK, int neighbor_width = NEIGHBORS_TO_CHECK_WIDTH, typename CheckBoidFunc>
__device__ void do_something_with_boid_neighbors(const glm::vec3 b_pos, int grid_resolution, float cell_width, const glm::vec3& grid_min, int* grid_cell_start_index, int* grid_cell_end_index, CheckBoidFunc check_boid_func)
{
  //offset boid position to [0:200]
  const glm::vec3 b_pos_offset = b_pos - grid_min;

  //truncate the float position into an int position
  glm::vec3 b_pos_int = b_pos_offset;
  truncate_glm_vec(b_pos_int);

  //this is the boid grid cell index (calculated by dividing by cell width)
  const glm::ivec3 b_cell_index = b_pos_offset / cell_width;

  //now get the position of the boid inside the cellWidth, which is cut up into 8
  const glm::vec3 b_pos_inside_cell_width = (b_pos_offset - b_pos_int) * cell_width;

  //find all the neighbors and populate neighbor array

  //which side is the grid on?
  enum class GridCellSide : uint8_t
  {
    Left,
    Right
  };

  //find which side are we on (point is either x, y, z)
  auto find_grid_cell_side = [&](float point)
  {
    return point < (cell_width / 2) ? GridCellSide::Left : GridCellSide::Right;
  };

  //find the side for x, y, z
  const GridCellSide x_side = find_grid_cell_side(b_pos_inside_cell_width.x);
  const GridCellSide y_side = find_grid_cell_side(b_pos_inside_cell_width.y);
  const GridCellSide z_side = find_grid_cell_side(b_pos_inside_cell_width.z);

  //find which side to iterate to (either -1 or 0)
  int x_offset = -neighbor_width + 1;
  int y_offset = -neighbor_width + 1;
  int z_offset = -neighbor_width + 1;

  if(x_side == GridCellSide::Right)
  {
    x_offset = 0;
  }

  if(y_side == GridCellSide::Right)
  {
    y_offset = 0;
  }

  if(z_side == GridCellSide::Right)
  {
    z_offset = 0;
  }

  //iterate x (either from -1 ... 0 or 0 ... 1)
  for(int i = x_offset; i < x_offset + neighbor_width; ++i)
  {
    const int x = b_cell_index.x + i;

    //check if out of bounds
    if(x < 0 || x >= grid_resolution)
    {
      continue;
    }

    //iterate y
    for(int k = y_offset; k < y_offset + neighbor_width; ++k)
    {
      const int y = b_cell_index.y + k;

      //check if out of bounds
      if (y < 0 || y >= grid_resolution)
      {
        continue;
      }

      //iterate z
      for(int l = z_offset; l < z_offset + neighbor_width; ++l)
      {
        const int z = b_cell_index.z + l;

        //check if out of bounds
        if (z < 0 || z >= grid_resolution)
        {
          continue;
        }

        //get the index into the grid_cell_start_index/grid_cell_end_inde
        const int index_into_grid_cell_pointer_index = gridIndex3Dto1D(x, y, z, grid_resolution);

        //compute the start and end indices for the grid cell index
        auto grid_start_index = grid_cell_start_index[index_into_grid_cell_pointer_index];
        const auto grid_end_index = grid_cell_end_index[index_into_grid_cell_pointer_index];

        //iterate through the boids in the grid cell
        for (; grid_start_index < grid_end_index; ++grid_start_index)
        {
          //pass in the boid array index
          check_boid_func(grid_start_index);
        }
      }
    }
  }
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

  const int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.

  //get the boid index
  const auto this_boid_index = particleArrayIndices[index];

  //get location of boid (float)
  const auto this_boid_pos = pos[this_boid_index];

  //do rule1, rule2, rule3
  int rule1_num_neighbors = 0;
  int rule3_num_neighbors = 0;

  glm::vec3 perceived_center = {};
  glm::vec3 c = {};
  glm::vec3 perceived_velocity = {};

  //for each neighbor, compute the rules and add them to perceived_center, c, perceived_velocity (rules 1, 2, 3)
  do_something_with_boid_neighbors(this_boid_pos,
                                   gridResolution, cellWidth,
                                   gridMin, gridCellStartIndices,
                                   gridCellEndIndices,
                                   [&](const int index_into_boid_array)
                                   {
                                     // - For each cell, read the start/end indices in the boid pointer array.
                                     // - Access each boid in the cell and compute velocity change from

                                     const int other_boid_index = particleArrayIndices[index_into_boid_array];
                                     check_rule1(this_boid_index, other_boid_index, pos,
                                                 perceived_center, rule1_num_neighbors);
                                     check_rule2(this_boid_index, other_boid_index, pos, c);
                                     check_rule3(this_boid_index, other_boid_index, pos,
                                                 perceived_velocity, rule3_num_neighbors, vel1);
                                   }
  );

  // - Clamp the speed change before putting the new speed in vel2

  //compute each rule
  const auto rule_1_result = finish_rule1(this_boid_pos, perceived_center, rule1_num_neighbors);
  const auto rule_2_result = finish_rule2(c);
  const auto rule_3_result = finish_rule3(perceived_velocity, rule3_num_neighbors);

  //add to velocity
  const auto vel = vel1[this_boid_index] + rule_1_result + rule_2_result + rule_3_result;
  vel2[this_boid_index] = clamp_vec3(vel);
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2

  const int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.

  //get the boid index
  const auto this_boid_index = index;

  //get location of boid (float)
  const auto this_boid_pos = pos[this_boid_index];

  //do rule1, rule2, rule3
  int rule1_num_neighbors = 0;
  int rule3_num_neighbors = 0;

  glm::vec3 perceived_center = {};
  glm::vec3 c = {};
  glm::vec3 perceived_velocity = {};

  //for each neighbor, compute the rules and add them to perceived_center, c, perceived_velocity (rules 1, 2, 3)
  do_something_with_boid_neighbors(this_boid_pos,
                                   gridResolution, cellWidth,
                                   gridMin, gridCellStartIndices,
                                   gridCellEndIndices,
                                   [&](const int index_into_boid_array)
                                   {
                                     // - For each cell, read the start/end indices in the boid pointer array.
                                     // - Access each boid in the cell and compute velocity change from

                                     const int other_boid_index = index_into_boid_array;
                                     check_rule1(this_boid_index, other_boid_index, pos,
                                                 perceived_center, rule1_num_neighbors);
                                     check_rule2(this_boid_index, other_boid_index, pos, c);
                                     check_rule3(this_boid_index, other_boid_index, pos,
                                                 perceived_velocity, rule3_num_neighbors, vel1);
                                   }
  );

  // - Clamp the speed change before putting the new speed in vel2

  //compute each rule
  const auto rule_1_result = finish_rule1(this_boid_pos, perceived_center, rule1_num_neighbors);
  const auto rule_2_result = finish_rule2(c);
  const auto rule_3_result = finish_rule3(perceived_velocity, rule3_num_neighbors);

  //add to velocity
  const auto vel = vel1[this_boid_index] + rule_1_result + rule_2_result + rule_3_result;
  vel2[this_boid_index] = clamp_vec3(vel);
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  // TODO-1.2 ping-pong the velocity buffers
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, dev_vel1, dev_vel2);
  checkCUDAError();
  
  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel1);
  checkCUDAError();

  swap_pointers(dev_vel2, dev_vel1);
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed

  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  dim3 cellPerGrid((gridCellCount + blockSize - 1) / blockSize);

  //reset the grid cell start / end
  kernResetIntBuffer<<<cellPerGrid, blockSize>>>(gridCellCount, dev_gridCellStartIndices, std::numeric_limits<int>::min());
  kernResetIntBuffer<<<cellPerGrid, blockSize>>>(gridCellCount, dev_gridCellEndIndices, std::numeric_limits<int>::min());

  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.

  //compute the indices
  kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  checkCUDAError();
  
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.

  //sort with thrust, so we have <key = grid index, value = array_index>
  thrust::device_ptr<int> thrust_grid_indices(dev_particleGridIndices);
  thrust::device_ptr<int> thrust_array_indices(dev_particleArrayIndices);

  thrust::sort_by_key(thrust_grid_indices, thrust_grid_indices + numObjects, thrust_array_indices);

  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices

  //make sure the start and end indices are mapped to the correct grid indices
  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
  checkCUDAError();

  // - Perform velocity updates using neighbor search

  //this finds possible neighboring particles in neighboring grid cells to find calculate the total velocity for a particle
  kernUpdateVelNeighborSearchScattered<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_particleArrayIndices, dev_pos, dev_vel1, dev_vel2);
  checkCUDAError();
  
  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel1);
  checkCUDAError();

  swap_pointers(dev_vel2, dev_vel1);
}

__global__ void kernSortPosAndVelByArrayIndicies(int N, glm::vec3* result_pos, glm::vec3* result_vel,
  glm::vec3* pos, glm::vec3* vel, int* grid_array)
{
  const int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }

  //fetch the pos and vel associated with the grid cell array
  const int pos_and_vel_index = grid_array[index];

  //copy over the pos/vel into the index that the grid array was in
  result_pos[index] = pos[pos_and_vel_index];
  result_vel[index] = vel[pos_and_vel_index];
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.

  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
  dim3 cellPerGrid((gridCellCount + blockSize - 1) / blockSize);

  //reset the grid cell start / end
  kernResetIntBuffer<<<cellPerGrid, blockSize>>>(gridCellCount, dev_gridCellStartIndices, std::numeric_limits<int>::min());
  kernResetIntBuffer<<<cellPerGrid, blockSize>>>(gridCellCount, dev_gridCellEndIndices, std::numeric_limits<int>::min());

  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.

  //compute the indices
  kernComputeIndices<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, dev_pos, dev_particleArrayIndices, dev_particleGridIndices);
  checkCUDAError();
  
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.

  //sort with thrust, so we have <key = grid index, value = array_index>
  thrust::device_ptr<int> thrust_grid_indices(dev_particleGridIndices);
  thrust::device_ptr<int> thrust_array_indices(dev_particleArrayIndices);

  thrust::sort_by_key(thrust_grid_indices, thrust_grid_indices + numObjects, thrust_array_indices);

  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices

  //sort the pos and vel indices by grid array
  kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_particleGridIndices, dev_gridCellStartIndices, dev_gridCellEndIndices);
  checkCUDAError();

  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.

  //make sure the start and end indices are mapped to the correct grid indices
  kernSortPosAndVelByArrayIndicies<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos_sorted, dev_vel_sorted, dev_pos, dev_vel1, dev_particleArrayIndices);
  checkCUDAError();

  // - Perform velocity updates using neighbor search

  //this finds possible neighboring particles in neighboring grid cells to find calculate the total velocity for a particle
  kernUpdateVelNeighborSearchCoherent<<<fullBlocksPerGrid, blockSize>>>(numObjects, gridSideCount, gridMinimum, gridInverseCellWidth, gridCellWidth, dev_gridCellStartIndices, dev_gridCellEndIndices, dev_pos_sorted, dev_vel_sorted, dev_vel2);
  checkCUDAError();

  //update position base on sorted pos/vel
  kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos_sorted, dev_vel_sorted);
  checkCUDAError();

  //ping pong
  swap_pointers(dev_vel2, dev_vel_sorted);
  swap_pointers(dev_vel2, dev_vel1);
  swap_pointers(dev_vel_sorted, dev_vel1);
  
  swap_pointers(dev_pos, dev_pos_sorted);
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  checkCUDAError();

  cudaFree(dev_vel2);
  checkCUDAError();

  cudaFree(dev_pos);
  checkCUDAError();

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.

  cudaFree(dev_particleArrayIndices);
  checkCUDAError();

  cudaFree(dev_particleGridIndices);
  checkCUDAError();

  cudaFree(dev_gridCellStartIndices);
  checkCUDAError();

  cudaFree(dev_gridCellEndIndices);
  checkCUDAError();

  //2.3 cleanup pos and vel struct
  cudaFree(dev_vel_sorted);
  checkCUDAError();

  cudaFree(dev_pos_sorted);
  checkCUDAError();
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAError();

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAError();

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  checkCUDAError();

  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  checkCUDAError();

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAError();

  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAError();

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  checkCUDAError();

  cudaFree(dev_intValues);
  checkCUDAError();
  return;
}

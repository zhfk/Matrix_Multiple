//¾ØÕóÏà³Ë
__kernel void MatrixMuti(
							__global int* restrict inputA,
							__global int* restrict inputB,
							int size,
							__global int* restrict output
						)
{ 
	int row = get_global_id(0);
	int col = get_global_id(1);
	int sum=0;
	for(int i=0;i<size;++i)
	{
		sum += inputA[row*size+i]*inputB[i*size+col]; 
	}
	output[row*size+col]=sum;
}
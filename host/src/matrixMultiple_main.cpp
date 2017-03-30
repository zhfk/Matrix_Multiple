/*
 * main.c
 *
 *  Created on: 2016年10月12日
 *      Author: zhfk
 */
//For clarity,error checking has been omitted.
//#pragma warning( disable : 4996 )

#include "tool.h"

using namespace std;
#define SIZE 20
#define RESULT_LENTH 20

int main(int argc, char* argv[])
{
	double start_time,end_time;
	cl_uint numOfDevice;
	cl_event events[2];
	cl_program program;
	cl_int    status;
	/**Step 1: Getting platforms and choose an available one(first).*/
	cl_platform_id platform;
	cout<<"program running---->"<<endl;
	getPlatform(platform);
	/**Step 2:Query the platform and choose the first GPU device if has one.*/
	cl_device_id *devices = getCl_device_id(platform,numOfDevice);

	/**Step 3: Create context.*/
	cl_context context = clCreateContext(NULL, 1, devices, NULL, NULL, NULL);

	/**Step 4: Creating command queue associate with the context.*/
	cl_command_queue commandQueue = clCreateCommandQueue(context, *devices, 0, NULL);

	//此处添加读取二进制 kernel文件
	  // Create the program for all device. Use the first device as the
	  // representative device (assuming all device are of the same type).

	const char *cl_kernel_file="CL_Kernel";
	string binary_file = getBoardBinaryFile(cl_kernel_file, devices[0]);
	printf("%-15s ===> %s \n","Using AOCX",binary_file.c_str());
	program= createProgramFromBinary(context, binary_file.c_str(), devices, numOfDevice);


	 // debug("Kernel execute scale calculate Matrix Multiple [%d x %d]",SIZE,SIZE);
	  // Build the program that was just created.
	status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
	// Shows the log
	ShowBuildLog(program, devices);

	/**Step 7: Initial input,output for the host and create memory objects for the kernel*/
	int* input_a = new int[SIZE*SIZE];
	int* input_b = new int[SIZE*SIZE];
	for (int i = 0; i < SIZE*SIZE; i++)
	{
		//input_a[i] = (rand()%20);
		//input_b[i] = (rand()%20);
		input_a[i] = i;
		input_b[i] = i;
	}
	int* output = new int[SIZE*SIZE];
	debug_msg(INFO,"Kernel calculate scale for Matrix Multiple [%dX%d]x[%dX%d]",SIZE,SIZE,SIZE,SIZE);
	debug_msg(INFO,"Kernel execute start...");
	start_time=getCurrentTimestamp();
	printf("%-15s ===> %.3f %s \n","start_time",start_time*1e3," Ms");
	//cout << "clCreateBuffer---------->" << endl << endl;
	cl_mem inputBuffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, SIZE*SIZE* sizeof(int), (void *)input_a, NULL);
	cl_mem inputBuffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, SIZE*SIZE* sizeof(int), (void *)input_b, NULL);
	cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZE*SIZE * sizeof(int), NULL, NULL);

	/**Step 8: Create kernel object */
	//cout << "clCreateKernel---------->" << endl << endl;
	cl_kernel kernel = clCreateKernel(program, "MatrixMuti", NULL);

	/**Step 9: Sets Kernel arguments.*/
	//cout << "clSetKernelArg---------->" << endl << endl;
	int matrixsize = SIZE;
	status  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&inputBuffer_a);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&inputBuffer_b);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&matrixsize);
	status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void *)&outputBuffer);

	/**Step 10: Running the kernel.*/
	//const size_t local_work_size[2] = { 32, 32 };
	const size_t global_work_size[2] = { SIZE, SIZE };

	//	const size_t local_ws = 512;    // Number of work-items per work-group
	//	cl_event enentPoint;
	//cout << "clEnqueueNDRangeKernel---------->" << endl << endl;
	status = clEnqueueNDRangeKernel(commandQueue, kernel, MatrixDim, NULL, global_work_size, NULL, 0, NULL, &events[0]);
	status = clWaitForEvents(1,&events[0]);
	if (status != CL_SUCCESS)
	{
		//cout <<"Error: Waiting for kernel run to finish.(clWaitForEvents)"<<endl;
		debug_msg(status, "Waiting for kernel run to finish.(clWaitForEvents)");
		return 0;
	}
	status = clReleaseEvent(events[0]);
	//将结果拷贝到主机端
	end_time = getCurrentTimestamp();
	//cout << "end_time :" << end_time*1e3<<" Ms" << endl;
	printf("%-15s ===> %.3f %s \n","end_time",end_time*1e3," Ms");
	//cout << "took time :" << ((end_time - start_time) * 1e3) << " Ms"<< endl;
	printf("%-15s ===> %.3f %s \n","took time",((end_time - start_time) * 1e3)," Ms");
	debug_msg(INFO,"Kernel execute finish !");

	/**Step 11: Read the cout put back to host memory.*/
	//cout << "clEnqueueReadBuffer---------->" << endl << endl;
	status = clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, SIZE*SIZE * sizeof(int), output, 0, NULL, &events[1]);
	status = clWaitForEvents(1, &events[1]);
	if (status != CL_SUCCESS)
	{
		//cout <<"Error: Waiting for read buffer call to finish. (clWaitForEvents)"<<endl;
		debug_msg(status, "Waiting for read buffer call to finish. (clWaitForEvents)");
		return 0;
	}

	status = clReleaseEvent(events[1]);
	debug_msg(INFO,"show %d result(s)",RESULT_LENTH);
	for (int i = 0; i < RESULT_LENTH; i++)
	{
		cout << input_a[i] << " ";
	}
	cout << endl;
	for (int i = 0; i < RESULT_LENTH; i++)
	{
		cout << input_b[i] << " ";
	}
	cout << endl;
	for (int i = 0; i < RESULT_LENTH; i++)
	{
		cout << output[i] << " ";
	}
	cout << endl;
	/**Step 12: Clean the resources.*/
	//cout << "clRelease---------->" << endl << endl;
	status = clReleaseKernel(kernel);//*Release kernel.
	status = clReleaseProgram(program);    //Release the program object.
	status = clReleaseMemObject(inputBuffer_a);//Release mem object.
	status = clReleaseMemObject(inputBuffer_b);//Release mem object.
	status = clReleaseMemObject(outputBuffer);
	status = clReleaseCommandQueue(commandQueue);//Release  Command queue.
	status = clReleaseContext(context);//Release context.

	if (input_a != NULL)
	{
		//cout << "clRelease---------->input_a" << endl << endl;
		delete[] input_a;
		input_a = NULL;
	}
	if (input_b != NULL)
	{
		//cout << "clRelease---------->input_b" << endl << endl;
		delete[] input_b;
		input_b = NULL;
	}
	if (output != NULL)
	{
		//cout << "clRelease---------->output" << endl << endl;
		delete[] output;
		output = NULL;
	}

	if (devices != NULL)
	{
		//cout << "clRelease---------->devices" << endl << endl;
		free(devices);
		devices = NULL;
	}
	cout << "program over--------->" << endl << endl;
	return 0;
}

//#include "stdafx.h"
#include<opencv2/opencv.hpp>
#include<iostream>
#include<algorithm>
#include<cmath>
#include"PriorityQueue.h"
//注：堆用的是索引优先队列，以及像素节点的数据结构均来自skeleton的代码，详见引用3
using namespace cv;
//节点状态
#define INITIAL 0
#define ACTIVE 1
#define EXPAND 2
//权值
#define WG 0.43f
#define WZ 0.43f
#define WD 0.14f
//Fz的缩放比例，决定抠图的精细程度，若让每个通道都小于15.0f，则会把一些非边缘点和边缘点混合，通过Fg来区分，这样就可以忽略一些细小的边缘
#define FzTrs 15.0f
//鼠标矫正的容错范围，访问越大，鼠标移动的位置可以约不精确，但是，同样会觉得鼠标不好操纵

//15.0f,适用于大致扣主要边缘的轮廓，1.0f以下是精细轮廓
struct PixelNode
{
	//像素节点对应像素点的坐标
	int col, row;
	//到八个邻接点的花费
	double linkCost[8];
	//该节点邻居节点的索引
	long long indexs[8];
	//该节点在数组中的索引
	long long thisIndex;
	//节点的状态
	int state;
	//种子点到这个节点的总花费
	double totalCost;
	// 前一个节点
	PixelNode *prevNode;
	//计时
	TickMeter tm;
	//记录显示时间和梯度值
	double countTime;
	//记录重绘次数
	double redraw;
	//节点初始化
	PixelNode() : linkCost{ 0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f },
		indexs{ -1,-1,-1,-1,-1 ,-1,-1,-1 },
		thisIndex(0),
		prevNode(nullptr),
		col(0),
		row(0),
		state(INITIAL),
		totalCost(0.0),
		countTime(0.0),
		redraw(0.0)
	{}
	// this function helps to locate neighboring node in node buffer. 
	void nbrNodeOffset(int &offsetX, int &offsetY, int linkIndex)
	{
		//计算与中心点的偏移量，也可作为位置向量之差  p-q

		/*
		*  321
		*  4 0
		*  567
		*/

		//在其他函数中会判断是否越界。

		if (linkIndex == 0)
		{
			offsetX = 1;
			offsetY = 0;
		}
		else if (linkIndex == 1)
		{
			offsetX = 1;
			offsetY = -1;
		}
		else if (linkIndex == 2)
		{
			offsetX = 0;
			offsetY = -1;
		}
		else if (linkIndex == 3)
		{
			offsetX = -1;
			offsetY = -1;
		}
		else if (linkIndex == 4)
		{
			offsetX = -1;
			offsetY = 0;
		}
		else if (linkIndex == 5)
		{
			offsetX = -1;
			offsetY = 1;
		}
		else if (linkIndex == 6)
		{
			offsetX = 0;
			offsetY = 1;
		}
		else if (linkIndex == 7)
		{
			offsetX = 1;
			offsetY = 1;
		}
	}
	//生成p-q，用于求像素点与邻居的位置向量
	cv::Vec2f genVector(int linkIndex)
	{
		int offsetX, offsetY;
		nbrNodeOffset(offsetX, offsetY, linkIndex);
		return cv::Vec2f(offsetX, offsetY);
	}
	// used by the binary heap operation, 
	// pqIndex is the index of this node in the heap.
	// you don't need to know about it to finish the assignment;
	int pqIndex;

	int &Index(void)
	{
		return pqIndex;
	}

	int Index(void) const
	{
		return pqIndex;
	}
};
//重载运算符，比较像素点的总花费大小
inline int operator < (const PixelNode &a, const PixelNode &b)
{
	return a.totalCost<b.totalCost;
}

class Scissor
{
private:
	//原图
	Mat src;
	//花费图
	Mat fzMat, fdMat, fgMat;
	//x方向偏导数，y方向偏导数
	Mat Dx, Dy;
	//是否设置种子
	bool seedflag ;  
	//原图的通道数；
	int channels;
	//原图一行的元素个数
	long long colNums;
	//像素总数
	long long totalnum;
	
protected:
	//laplacian算子
	Mat laplacian = (Mat_<char>(3, 3) <<
		0, -1, 0,
		-1, 4, -1,
		0, -1, 0);
	//用scharr滤波器代替sobel效果更好,自带平滑和卷积，就不用gaussianBlur
	Mat scharrx = (Mat_<char>(3, 3) <<
		-3, 0, 3,
		-10, 0, 10,
		-3, 0, 3);//对像素求x偏微分

	Mat scharry = (Mat_<char>(3, 3) <<
		-3, -10, -3,
		0, 0, 0,
		3, 10, 3);//对像素求y偏微分

 
public:
		 Mat oldFgMat;
	    //原图对应节点数
	     PixelNode * nodes;
	     Scissor();
		 Scissor(Mat& origin);
		 //是否设置种子点
		 bool isSeed();
		 //卷积通用函数，调用OpenCV自带的filter2D，单kernel的大小>=11时，会调用DFT，提高效率。
		 //会自动扩充边界
		 void filter(const Mat& origin, Mat& dst, Mat& kernel);
		 //计算偏导数
		 void calculateDxDy();
		 //计算Fz,选择每个通道都同时小于阈值的点，花费为零
		 void calculateFz();
		 //选择三个通道中梯度值最大的点，梯度值越大花费越低
		 void calculateFg();
		 //求Lpq，p_q是位置矢量，Dp,用来判断方向，Lpq是结果
		 void getLpq(Vec2f&p_q, Vec2f&Dp, Vec2f&Lpq);
		 //计算Fd,用三个通道的x方向的偏导数之和作为总的，x方向的偏导数，y方向同理，然后求出每一个像素点的单位方向梯度矢量，存在一个Mat中
		 void calculateFd();
		 //累加Fz,Fg,并乘以相应的权重
		 void accumulateCost();
		 //求种子点到该点的最短路径
		 void liveWire(long long index);//const int & col, const int & row
		 //设置seedFlag
		 void setSeed(bool flag);
		 //计算一幅图的总像素数
		 long long getTotalNums();
		 //选用每个节点到邻居最短的费用作为这个点的花费，再乘以255 作为该点的像素值
		 void showGray();

		 void calculateFg(Mat& myMat);

		 long long cursorSnap(long long index);
		 virtual ~Scissor();
};


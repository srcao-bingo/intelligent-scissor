#include "stdafx.h"
#include "Scissor.h"
#include<algorithm>
const double SQRT2 = 1.4142;
const double invSqrt2 = 1.0 / SQRT2;
const double PI = 3.1415926;
const double invPi = 4.0 /( 3.0 * PI);
//const double _2Over3PI = 2.0 / (3.0 * PI);
extern int CURSERSNAPTRS = 10;
using namespace cv;
Scissor::Scissor()
{
}

Scissor::Scissor(Mat& origin)
{
	seedflag = false;

	src = origin.clone();

	totalnum = src.cols*src.rows;

	int& row = src.rows;

	int& col = src.cols;

	channels = src.channels();

	colNums = channels * src.cols;

	fzMat.create(src.size(), CV_32FC1);

	fgMat.create(src.size(), CV_32FC1);

	fdMat.create(src.size(), CV_32FC1);
	//每一个像素点对应一个像素节点
	nodes = new PixelNode[totalnum+10];

	int index = 0;
	//初始化每个节点
	for(int i = 0;i < row ; i++)
		for (int j = 0; j < col; j++)
		{
			index = i * col + j;
			nodes[index].row = i;
			nodes[index].col = j;
			nodes[index].totalCost = 0.0f;
			nodes[index].thisIndex = index;
			//其余变量再分配空间的时候已经初始化好了
		}
	//计算偏导数
	calculateDxDy();
	calculateFz();
	calculateFg();
	calculateFd();
	accumulateCost();
	//liveWire(150);
}

void Scissor::filter(const Mat& origin, Mat& dst, Mat& kernel)
{
	//把卷积后的数据转换成32位提高精度，filter会对多通道，分通道处理
	filter2D(origin, dst, CV_32F, kernel);
}

void Scissor::calculateDxDy()
{
	filter(src, Dx, scharrx);

	filter(src, Dy, scharry);
}
//modified
void Scissor::calculateFz()
{
	int type = 0;
	//对应单通道，三通道生成不同的Mat
	if (channels == 1)
	{
		type = CV_32FC1;
	}
	else {
		type = CV_32FC3;
	}

	Mat tmpFz(src.rows, src.cols, type);
	//存放高斯滤波过后的图
	Mat afterBlur(src.rows, src.cols, src.type());

	Size mysize;
	//用5*5的高斯核
	mysize.width = mysize.height =  5;

	//因为导数对噪点敏感，先用高斯滤波降噪 !!!
	GaussianBlur(src, afterBlur,mysize , 0, 0);

	filter(afterBlur, tmpFz, laplacian);//--------afterBlur----------------

	float* p, *q;

	int count = 0;
	
	for (int i = 0; i < tmpFz.rows; i++)
	{
		p = tmpFz.ptr<float>(i);

		q = fzMat.ptr<float>(i);

		//因为边缘采用laplacian算子卷积后，三个通道的二阶导数都小于零，所以直接找三个二阶导数都小于零的像素点，就是边缘点
		for (int j = 0; j < colNums - 2; j += 3)//车身不完整？？  
											 //j = j + 3;佩服的五体投地，3个像素，调了半个多小时！！！
		{
			
			if (p[j] < FzTrs && p[j + 1] < FzTrs && p[j + 2] < FzTrs)
			{
				q[count++] = 0;
			}
			else
			{
				q[count++] = 1;
			}
		}
		//还原
		count = 0;
	}
	p = q = NULL;
}
//modified
void Scissor::calculateFg()
{
	float* p;
	float* q;
	float* g;
	float gMax = -1.0f;
	float gMin = 6553666.0f;

	int type = 0;
	//对应单通道，三通道生成不同的Mat
	if (channels == 1)
	{
		type = CV_32FC1;
	}
	else {
		type = CV_32FC3;
	}
	//存放三个通道的梯度幅值
	Mat tmpFg(src.rows, src.cols, type);
	//求梯度幅值
	for (int i = 0; i < Dx.rows; i++)
	{
		p = Dx.ptr<float>(i);
		q = Dy.ptr<float>(i);
		g = tmpFg.ptr<float>(i);
		for (int j = 0; j < colNums; j++)
		{
			g[j] = sqrt(p[j] * p[j] + q[j] * q[j]);

		}

	}
	//记录原来的梯度幅值，用于路径冷却。
	oldFgMat = tmpFg.clone();

	int count = 0;
	for (int i = 0; i < tmpFg.rows; i++)
	{
		p = tmpFg.ptr<float>(i);

		g = fgMat.ptr<float>(i);

		//选取每个通道的最大值,效果最好
		for (int j = 0; j < colNums - 2; j += 3)
		{
			
			//g[count] = (p[j] + p[j + 1] + p[j + 2]);///3.0f;//
			g[count] = max({ p[j],p[j + 1],p[j + 2] });

			if (g[count]>gMax)
			{
				gMax = g[count];
			}
			if (g[count] < gMin)
			{
				gMin = g[count];
			}

			count++;
		}

		count = 0;
	}

	float coefG = gMax - gMin;

	for (int i = 0; i < fgMat.rows; i++)
	{
		g = fgMat.ptr<float>(i);

		//梯度幅值越大，边缘的可能性越大，费用越低
		for (int j = 0; j < fgMat.cols; j++)
		{
			g[j] = 1.0f - (g[j] - gMin) / coefG;
		}

	}

	p = q = g = NULL;

	//注意后面累加的时候，对角线要除以sqrt(2);
}
//
void Scissor::calculateFd()//思路必须特别顺！！！
{
	//先求每一个通道的梯度方向的单位法矢量，并且存到一个Mat里，避免重复求取
	//因为是二维矢量，所以用两通道的Mat存储
	Mat tmpFd(src.rows, src.cols, CV_32FC2);
	int count = 0 ;
	float* dx;
	float* dy;
	Vec2f* p , *q;
	/*if (channels == 1)
	{
		for (int i = 0; i < tmpFd.rows; i++)
		{
			count = 0;
			dx = Dx.ptr<float>(i);
			dy = Dy.ptr<float>(i);
			p = tmpFd.ptr<Vec2f>(i);//无敌心痛的Vec2f,破书一句话带过……，按vec2f的方式访问,就是同时访问两个通道
			for (int j = 0; j < colNums; j++)
			{
				//归一化
				p[count] = normalize(Vec2f(dy[j], -dx[j]));

				count++;
			}
		}
	}*/
	 if(channels == 3)
	{		
		float sumx;
		float sumy;
		for (int i = 0; i < tmpFd.rows; i++)
		{
			count = 0;
			dx = Dx.ptr<float>(i);
			dy = Dy.ptr<float>(i);
			p = tmpFd.ptr<Vec2f>(i);//无敌心痛的Vec2f,破书一句话带过……1.5h
			for (int j = 0; j < colNums - 2; j += 3)
			{
				//采用CSDN博客三的处理方式，将三个通道的x方向偏导数值相加
				//因为三个法矢量之和对轮廓的方向改变描绘清晰，且运算快，所以使用累加
				sumx = (dx[j] + dx[j + 1] + dx[j + 2]);
				
				sumy = (dy[j] + dy[j + 1] + dy[j + 2]);
				
				 p[count] = normalize(Vec2f(sumy, -sumx));

				count++;
			}
		}

	}

	dx = dy = NULL;

	Vec2f q_p, Lpq,tmpSum;//p-q , Lpq
	long long index = 0;
	int offSetRow, offSetCol;
	float dp,dq,tmp;
	//利用genVector以及nodes自带的函数，生成p-q的位置向量，即与邻居相对位置的向量。


	//！！！边界元素不考虑，即最外面一圈不考虑，因为最外一圈是边界只有四个转角点，Fd不为零，可以忽略不计！！！！！
	for (int i = 1; i < tmpFd.rows-1; i++)
	{
		for (int j = 1; j < tmpFd.cols-1; j++)
		{
			//得到单位法矢量
			p = tmpFd.ptr<Vec2f>(i,j);
			//找到对应像素节点的下标,不能直接index++,因为边缘的默认为零
			index = i * tmpFd.cols + j;
			
			for (int k = 0; k < 8; k++)
			{
				
				//得到 Q - P 向量形式（Vec2f） 
				q_p = nodes[index].genVector(k);
				//对每一个邻居求一遍Lpq,并归一化
				getLpq(q_p, (*p), Lpq);
				//注意nbrNodeOffset，返回的是列、行，别弄反了！！！！！！！！！！！
				nodes[index].nbrNodeOffset(offSetCol,offSetRow,k);
				
				q = tmpFd.ptr<Vec2f>(i + offSetRow, j + offSetCol);
				//求dp(p,q) dq(p,q)并计算fD

				tmpSum = normalize(*p + *q);

				tmp = tmpSum.dot(Lpq);

				//dp = p->dot(Lpq);

				//dq = q->dot(Lpq);
				//----具体实现时，我将两个单位法向量相加，再归一化，再求相加后的向量与Lpq之间的夹角，因为此时，最大的夹角为3.0*pi/4,所以归一化的系数是4/（3.0*pi）
				nodes[index].linkCost[k] = invPi *(acos(tmp))*WD;//--_2Over3PI *(acos(dp)+acos(dq))*WD ---------------invPi *(acos(tmp))*WD--------------


			}
			
		}
		
	}

	p = q = NULL;

}
void Scissor::getLpq(Vec2f&p_q,Vec2f&Dp,Vec2f&Lpq)
{
	float invP_Q = 1.0f / sqrt(p_q.dot(p_q));

	if (Dp.dot(p_q) >= 0.0f)
	{
		Lpq = invP_Q * p_q;
	}
	else
	{
		Lpq = -1.0f * invP_Q * p_q;
	}

}
//累加到邻居上,以优化
void Scissor::accumulateCost()
{
	long long index = 0;
	int offSetRow, offSetCol ,i,j,k;
	int rows = fzMat.rows, cols = fzMat.cols;
	int neighborRow, neighborCol;
	//long long neighborIndex;//顺便把邻居的索引也记录下来，这叫做索引节点，为后面索引优先队列服务
	float *p, *q;
	for ( i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			//index = i * cols + j;

			for (k = 0; k < 8; k++)
			{
				//先返回列再返回行………………
				nodes[index].nbrNodeOffset(offSetCol, offSetRow, k);
				neighborRow = i + offSetRow;
				neighborCol = j + offSetCol;
				//越界检查
				if (neighborCol < 0 || neighborCol >= cols || neighborRow < 0 || neighborRow >= rows)
				{
					continue;
				}
				p = fzMat.ptr<float>(neighborRow, neighborCol);

				q = fgMat.ptr<float>(neighborRow, neighborCol);

				//neighborIndex = 
				//采用了论文二的方式为，让直接相邻的Fg花费除以sqrt(2);
				if (k % 2 == 0)
				{
					nodes[index].linkCost[k] += (*p)*WZ + (*q)*WG*invSqrt2;//--------------------------论文二
				}
				else
				{
					nodes[index].linkCost[k] += (*p)*WZ + (*q)*WG;
				}

				nodes[index].indexs[k] = neighborRow * cols + neighborCol;
			}

			index++;
		}
	}
}
//pixelNode数据结构修改，查找比他快，整体对堆的使用基本一致。索引优先队列，知道怎么用就行了。已优化，还有一个tmpIndex赋值便于阅读，就不在简化了
void Scissor::liveWire(long long index)//const int & col , const int & row
{

	//局部初始化，就是先把邻居初始化，再从邻居里找，花费最小的，
	//如果有邻居已经初始化好了，就比较种子点通过这个点到它的邻居和种子点通过其他点到它的距离
	//如果花费更小就更新updata(r);
	//直到堆空了，说明所有点都已经被扩展了。
	//以下全是我自己按照之前的想法以及CSDN第三个博客的部分印象写的

	//鼠标输入的 X = COL  Y = ROW
	//int index = row * src.cols + col;
	//int offSetRow, offSetCol, neiBorRow, neiBorCol;

	CTypedPtrHeap<PixelNode> pq;

	int i = 0;

	long long tmpIndex;

	PixelNode * node = (nodes+index);

	node->totalCost = 0.0f;

	node->prevNode = NULL;

	seedflag = true;

	pq.Insert(node);

	while (!pq.IsEmpty())
	{
		//从堆顶取出最小的元素
		node = pq.ExtractMin();
		node->state = EXPAND;
		for (i = 0; i < 8; i++)
		{
			tmpIndex = node->indexs[i];

			if (tmpIndex == -1 || nodes[tmpIndex].state == EXPAND)
			{
				continue;
			}
			
			else if (nodes[tmpIndex].state == INITIAL)
			{
				nodes[tmpIndex].totalCost = node->totalCost + node->linkCost[i];

				nodes[tmpIndex].state = ACTIVE;

				nodes[tmpIndex].prevNode = node;

				pq.Insert(&nodes[tmpIndex]);
			}
			else if (nodes[tmpIndex].state == ACTIVE)
			{
				if (nodes[tmpIndex].totalCost > node->totalCost + node->linkCost[i])
				{
					nodes[tmpIndex].totalCost = node->totalCost + node->linkCost[i];

					nodes[tmpIndex].prevNode = node;

					pq.Update(&nodes[tmpIndex]);

				}

			}
			
		}

	}

	//因为每一次点击都会重复使用nodes,所以对nodes的标志必须重置
	//-----------------------------------------------------------------------
	for (i = 0; i < totalnum; i++)
	{
		nodes[i].state = INITIAL;
	}

	//-----------------------------------------------------------------------


}

void Scissor::setSeed(bool flag)
{
	seedflag = flag;
}

void Scissor::showGray()
{
	Mat Gray = Mat(src.rows, src.cols, CV_8UC1);

	double pixVal = 0;

	for (int i = 0; i < totalnum; i++)
	{
		pixVal = min({ nodes[i].linkCost[0],nodes[i].linkCost[1], nodes[i].linkCost[2], nodes[i].linkCost[3], nodes[i].linkCost[4], nodes[i].linkCost[5],nodes[i].linkCost[6],nodes[i].linkCost[7] });
	
		*Gray.ptr<uchar>(nodes[i].row, nodes[i].col) = pixVal * 255.0;
	}

	imshow("Unequalized", Gray);

	Mat tmpGray = Gray.clone();

	//直方图均衡化，增加对比度。
	equalizeHist(Gray, tmpGray);

	imshow("equalizeHistGray", tmpGray);
	//gama增强
	double coef = 1.0 / (100.0),gama = 0.6;

	for (int i = 0; i < tmpGray.rows; i++)
	{
		for (int j = 0; j < tmpGray.cols; j++)
		{
			*tmpGray.ptr<uchar>(i, j) *= coef;
			*tmpGray.ptr<uchar>(i, j) = pow(*tmpGray.ptr<uchar>(i,j), gama);
			*tmpGray.ptr<uchar>(i,j) *= 255;
		}
	}
	
	imshow("gamaGray", tmpGray);
	
	calculateFg(Gray);
	
	imshow("magnitude", Gray);

	tmpGray.~Mat();

	Gray.~Mat();
}
//光标矫正
long long Scissor::cursorSnap(long long index) 
{
	if (index <0 || index >= totalnum)
	{
		std::cout << "鼠标坐标越界" << std::endl;
		return 0;
	}

	float MinFg, *p;
	
	long long maxIndex = index;
	
	int tmpRow = nodes[index].row, tmpCol = nodes[index].col;
	
	p = fgMat.ptr<float>(tmpRow, tmpCol);

	MinFg = *p;

	for (int i = nodes[index].row - CURSERSNAPTRS; i < nodes[index].row + CURSERSNAPTRS; i++)
	{
		if (i < 0 || i >= fgMat.rows)
		{
			continue;
		}
		for (int j = nodes[index].col - CURSERSNAPTRS; j < nodes[index].col + CURSERSNAPTRS; j++)
		{
			if (j < 0 || j >= fgMat.cols)
			{
				continue;
			}

			p = fgMat.ptr<float>(i,j);

			if (*p < MinFg)
			{
				tmpRow = i;
				tmpCol = j;
				MinFg = *p;
			}

		}

	}

	return tmpRow*fgMat.cols + tmpCol;

}

void Scissor::calculateFg(Mat& myMat)
{
	float* p;
	float* q;
	float* g;
	Mat dx, dy;

	filter(myMat, dx, scharrx);

	filter(myMat, dy, scharry);
	
	int	type = CV_32FC1;
	
	//存放受理后的灰度值
	Mat tmpFg(myMat.rows, myMat.cols, type);
	//求梯度幅值
	for (int i = 0; i < myMat.rows; i++)
	{
		p = dx.ptr<float>(i);
		q = dy.ptr<float>(i);
		g = tmpFg.ptr<float>(i);
		for (int j = 0; j < myMat.cols; j++)
		{
			//直接缩放除以255.0
			g[j] = sqrt(p[j] * p[j] + q[j] * q[j])/255.0;
		}

	}
	
	for (int i = 0; i < tmpFg.rows; i++)
	{
		g = tmpFg.ptr<float>(i);

		//梯度幅值越大，边缘的可能性越大，费用越低
		for (int j = 0; j < tmpFg.cols; j++)
		{
			//简单的二值化，目的是让边缘为黑色，非边缘为白色
			if (g[j] >= 1.0)
			{
				g[j] = 0.0;
			}
			else {
				g[j] = 1.0 - g[j];
			}
		}

	}
	tmpFg.copyTo(myMat);
	//waitKey(0);
	p = q = g = NULL;

	dx.~Mat();

	dy.~Mat();

	tmpFg.~Mat();
}

long long Scissor::getTotalNums()
{
	return totalnum;
}

bool Scissor::isSeed()
{
	return seedflag;

}

Scissor::~Scissor()
{
}

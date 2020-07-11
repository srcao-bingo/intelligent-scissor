#ifndef PRIORITYQUEUE_H
#define PRIORITYQUEUE_H

template < class T >
class CTypedPtrArray
{
protected:
	//堆指针的头
	T * *m_pHead;
	//堆的大小
	int m_iSize;
	//堆尾的下标
	int m_iTail;
public:
	CTypedPtrArray(void);
	~CTypedPtrArray(void);

	bool IsEmpty(void) const { return m_iTail == 0; }//堆尾的下标为零，表示堆空
	int GetSize(void) const { return m_iTail; }
	int GetTrueSize(void) const { return m_iSize; }
	bool IsIndexValid(const int i) const { return (i >= 0) && (i < m_iTail); }

	void SetTailAt(const int i) { m_iTail = i; }//设置堆尾
	void SetSize(int);

	int AddTail(T *);
	bool RemoveTail(void);
	void RemoveAll(void);
	bool SetAt(int, T *);

	T *&operator[] (const int index) { return m_pHead[index]; }
	T *ElementAt(const int index) const { return m_pHead[index]; }
	void FreePtrs(void);
};

template < class T >
CTypedPtrArray< T >::CTypedPtrArray(void)
{
	m_pHead = NULL;
	m_iSize = 0;
	m_iTail = 0;
}

template < class T >
CTypedPtrArray< T >::~CTypedPtrArray(void)
{
	if (m_pHead)
		delete[] m_pHead;
}

template < class T >
void CTypedPtrArray< T >::SetSize(const int size)
{
	if (size == 0)//置零表示全删
	{
		RemoveAll();
		return;
	}

	T **oldArray = m_pHead;
	int oldSize = m_iSize;

	m_iSize = size;
	if (m_iTail > m_iSize)
		m_iTail = m_iSize;
	m_pHead = new T*[m_iSize];
	// Fixed by Loren. Was:
	// memset( m_pHead, NULL, sizeof( T * ) * m_iSize );
	memset(m_pHead, 0, sizeof(T *) * m_iSize);

	if (oldSize)
	{
		if (m_iTail)
			memcpy(m_pHead, oldArray, sizeof(T *) * m_iTail);
		delete[] oldArray;
	}
}

template < class T >
int CTypedPtrArray< T >::AddTail(T *t)
{
	if (m_iTail == m_iSize)
		SetSize(m_iSize ? m_iSize << 1 : 1);

	m_pHead[m_iTail] = t;
	int index = m_iTail;
	m_iTail++;
	return index;
}

template < class T >
bool CTypedPtrArray< T >::RemoveTail(void)
{
	if (IsEmpty())
		return false;

	m_iTail--;
	m_pHead[m_iTail] = NULL;
	if (m_iTail <= (m_iSize >> 2))//可能和二叉堆的大小有关
		SetSize(m_iSize >> 1);
	return true;
}

template < class T >
void CTypedPtrArray< T >::RemoveAll(void)
{
	if (m_pHead)
	{
		delete[] m_pHead;
		m_pHead = NULL;
		m_iSize = 0;
		m_iTail = 0;
	}
}

template < class T >
bool CTypedPtrArray< T >::SetAt(int index, T *t)
{
	if (IsIndexValid(index))
	{
		m_pHead[index] = t;
		return true;
	}
	else
		return false;
}

template < class T >
void CTypedPtrArray< T >::FreePtrs(void)
{
	int i;
	for (i = 0; i < m_iSize; i++)
	{
		if (m_pHead[i])
		{
			delete m_pHead[i];
			m_pHead[i] = NULL;
		}
	}
}

/////////////////////
//definition of typed pointer Heap based on typed pointer array;

template < class T >
class CTypedPtrHeap : public CTypedPtrArray< T >
{
public:
	CTypedPtrHeap(void) : CTypedPtrArray< T >() {}
	~CTypedPtrHeap(void) {}

private:
	//
	void Heapify(const int index);
	//上浮
	void BubbleUp(const int index);

public:
	static int Parent(int i) { ++i; i >>= 1; return --i; }
	static int Left(int i) { ++i; i <<= 1; return --i; }
	static int Right(int i) { ++i; i <<= 1; return i; }

	void Insert(T *t);
	bool RemoveAt(const int index);
	bool Remove(const T *t);
	bool UpdateAt(const int index);
	bool Update(const T *t);
	void BuildHeap(void);
	T *ExtractMin(void);
};

template < class T >
//堆化，直接把数组转换成最小堆，比直接建堆快
void CTypedPtrHeap< T >::Heapify(const int index)
{
	if (!CTypedPtrArray<T>::IsIndexValid(index))
		return;

	int i = index;
	while (true) {
		int least = i;

		int left = Left(i);
		if (left < CTypedPtrArray<T>::m_iTail &&
			*(CTypedPtrArray<T>::m_pHead[left]) < *(CTypedPtrArray<T>::m_pHead[least]))
			least = left;

		int right = Right(i);
		if (right < CTypedPtrArray<T>::m_iTail &&
			*(CTypedPtrArray<T>::m_pHead[right]) < *(CTypedPtrArray<T>::m_pHead[least]))
			least = right;

		if (least != i) {
			T *t = CTypedPtrArray<T>::m_pHead[least];
			CTypedPtrArray<T>::m_pHead[least] = CTypedPtrArray<T>::m_pHead[i];
			CTypedPtrArray<T>::m_pHead[least]->Index() = least;
			CTypedPtrArray<T>::m_pHead[i] = t;
			CTypedPtrArray<T>::m_pHead[i]->Index() = i;
			i = least;
		}
		else {
			break;
		}
	}
}

template < class T >
void CTypedPtrHeap< T >::BubbleUp(const int index)
{
	if (!CTypedPtrArray<T>::IsIndexValid(index))
		return;

	int i = index;
	T *t = CTypedPtrArray<T>::m_pHead[index];

	while (Parent(i) >= 0 && *t < *(CTypedPtrArray<T>::m_pHead[Parent(i)])) {
		CTypedPtrArray<T>::m_pHead[i] = CTypedPtrArray<T>::m_pHead[Parent(i)];
		CTypedPtrArray<T>::m_pHead[i]->Index() = i;
		i = Parent(i);
	}
	CTypedPtrArray<T>::m_pHead[i] = t;
	CTypedPtrArray<T>::m_pHead[i]->Index() = i;
}

template < class T >
void CTypedPtrHeap< T >::Insert(T *t)
{
	int index = CTypedPtrHeap< T >::AddTail(t);
	BubbleUp(index);
}

template < class T >
bool CTypedPtrHeap< T >::RemoveAt(const int index)
{
	if (!CTypedPtrArray<T>::IsIndexValid(index))
		return false;

	if (index == CTypedPtrArray<T>::m_iTail - 1)
		CTypedPtrArray<T>::RemoveTail();
	else {
		CTypedPtrArray<T>::m_pHead[index] = CTypedPtrArray<T>::m_pHead[CTypedPtrArray<T>::m_iTail - 1];
		CTypedPtrArray<T>::m_pHead[index]->Index() = index;
		CTypedPtrArray<T>::RemoveTail();
		UpdateAt(index);
	}
	return true;
}

template < class T >
bool CTypedPtrHeap< T >::Remove(const T *t)
{
	return RemoveAt(t->Index());
}

template < class T >
bool CTypedPtrHeap< T >::UpdateAt(const int index)
{
	if (!CTypedPtrArray<T>::IsIndexValid(index))
		return false;

	T *t = CTypedPtrArray<T>::m_pHead[index];
	Heapify(index);

	if (t == CTypedPtrArray<T>::m_pHead[index])
		BubbleUp(index);

	return true;
}

template < class T >
bool CTypedPtrHeap< T >::Update(const T *t)
{
	return UpdateAt(t->Index());
}

template < class T >
void CTypedPtrHeap< T >::BuildHeap(void)
{
	if (CTypedPtrArray<T>::IsEmpty())
		return;

	for (int i = Parent(CTypedPtrArray<T>::m_iTail - 1); i >= 0; i--)
		Heapify(i);
}

template < class T >
T *CTypedPtrHeap< T >::ExtractMin(void)
{
	if (CTypedPtrArray<T>::IsEmpty())
		return NULL;

	T *t = CTypedPtrArray<T>::m_pHead[0];
	RemoveAt(0);
	return t;
}



#endif

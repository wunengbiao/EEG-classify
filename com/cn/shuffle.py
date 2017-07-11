def shuffle(a,k):
    mid=int(len(a)/2)
    left=a[0:mid]
    right=a[mid:]
    result=[]

    if k==0:
        return a

    for j in range(mid):
        result.append(right[mid-j-1])
        result.append(left[mid-j-1])
    result.reverse()
    return shuffle(result,k-1)


if __name__=='__main__':
    inputs=[]
    T=int(input())
    K=[]
    for i in range(T):
        n,k=input().split(' ')
        tmp=[i for i in input().split(' ')]
        inputs.append(tmp)
        K.append(int(k))

    for i in range(T):
        result=shuffle(inputs[i],K[i])
        for i in result:
            print(i,end=' ')
        print()
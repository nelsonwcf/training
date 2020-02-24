## solution for arrays of comparable sizes
# time: O(N+M), space: O(min(N,M))
def find_duplicates_v1(arr1, arr2):
    out = []
    i1 = 0
    i2 = 0
    
    while i1 < len(arr1) and i2 < len(arr2):
        if arr1[i1] == arr2[i2]:
            out.append(arr1[i1])
            i1 += 1
            i2 += 1            
        elif arr1[i1] > arr2[i2]:
            i2 += 1
        else:
            i1 += 1

    return out

## solution in which one array is much greater than the other
## time: O(N.Log(M)), space: O(min(N,M))
def find_duplicates_v2(arr1, arr2):
    out = []
    i1 = 0
    i2 = 0
    
    if len(arr1) > len(arr2):
        x = arr1
        arr1 = arr2
        arr2 = x
    
    while i1 < len(arr1) and i2 < len(arr2):
        if arr1[i1] == arr2[i2]:
            out.append(arr1[i1])
            i1 += 1
            i2 += 1            
        elif arr1[i1] > arr2[i2]:
            i2 = next_greater_(arr2, arr1[i1], 0, len(arr2)-1)
        else:
            i1 += 1
            
    return out
  
def next_greater_(arr, n, left, right):
    if left > right:
        return left
    
    mid = int((left + right) / 2)
    
    if arr[mid] == n:
        return mid
    elif n < arr[mid]:
        return next_greater_(arr, n, left, mid - 1)
    else:
        return next_greater_(arr, n, mid + 1, right)
      
# chosing which version to use
def find_duplicates(arr1, arr2):
  return find_duplicates_v2(arr1, arr2)
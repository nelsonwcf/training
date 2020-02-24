def reverse_words(arr):
    if len(arr) <= 1:
        return arr

    # in-place reversal
    mirror_(arr, 0, len(arr) - 1)

    start_position = 0
    for i in range(len(arr)):
        if arr[i] == ' ':
            mirror_(arr, start_position, i - 1)
            start_position = i + 1

    mirror_(arr, start_position, len(arr) - 1)
    
    return arr

def mirror_(arr, i, j):
    while i <= j:
        c = arr[i]
        arr[i] = arr[j]
        arr[j] = c
        i += 1
        j -= 1

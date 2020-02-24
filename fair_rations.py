def fair_rations(B):
    t=0
    for i in range(len(B)-1):
        if B[i] % 2 == 1:
            t += 2
            B[i] += 1
            B[i+1] += 1
    if B[-1] % 2 == 0:
        return t

    return 'NO'

if __name__ == '__main__':
    N = int(input())
    B = list(map(int, input().rstrip().split()))
    print(fairRation(B))


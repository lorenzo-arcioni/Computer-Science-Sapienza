def solve():
    import sys
    data = sys.stdin.read().strip().split()
    t = int(data[0])
    index = 1
    out_lines = []
    for _ in range(t):
        n = int(data[index]); index += 1
        # a[1..n] (notare che a[0] e a[n+1] saranno usati come “sentinella” pari a zero)
        a = [0]*(n+2)
        for i in range(1, n+1):
            a[i] = int(data[index]); index += 1
        L = [0]*(n+2)
        R = [0]*(n+2)
        L[0] = 0
        for i in range(1, n+1):
            L[i] = (a[i] + L[i-1]) // 2
        R[n+1] = 0
        for i in range(n, 0, -1):
            R[i] = (a[i] + R[i+1]) // 2
        ans = []
        for i in range(1, n+1):
            val = a[i] + L[i-1] + R[i+1]
            ans.append(str(val))
        out_lines.append(" ".join(ans))
    sys.stdout.write("\n".join(out_lines))
 
if __name__ == '__main__':
    solve()
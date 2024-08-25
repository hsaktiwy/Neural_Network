c  = int(input())
a = []
for i in range(c):
    a.append(int(input()))
v = int(input())
b = []
for i in range(v):
    b.append(int(input()))
res=''
cout  = 0
for i in range(c):
    if not (a[i] in b):
        res +=str(a[i]) + " " if i != c - 1 else ""
        cout+=1

print(cout)
print(res)
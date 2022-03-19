from eight_queens import view_solutions

solutions = []
temp = []
f = open('solutions.txt')
for i in f:
    temp.append(i)
f.close()

for i in range(len(temp)):
    arr = temp[i].strip('][\n').split(", ")
    for j in range(len(arr)):
        arr[j] = int(arr[j])
    solutions.append(arr)

view_solutions(solutions)

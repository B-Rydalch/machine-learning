import matplotlib.pyplot as plt

x = [1,2,3]
y = [5,7,4]

#plt.plot(x,y)
#plt.scatter(x, y)
plt.bar([2,4,6], [5,7,4],label = "bar1", color = "red")
plt.bar([1, 3, 5], [7, 8, 2], label = "bar2")
plt.xlabel("Semester Time period")
plt.ylabel("Machine learning difficulty")
plt.title("python tip of the day")


plt.show()

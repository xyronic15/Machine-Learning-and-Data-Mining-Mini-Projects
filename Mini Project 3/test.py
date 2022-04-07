from sklearn.datasets import load_digits
digits=load_digits()

x=digits.images.reshape((len(digits.images),-1))
print(x.shape)
y=digits.target
print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train.shape)
print(type(X_train.T))

y_train = y_train.reshape(1, y_train.shape[0])
print(y_train.shape)
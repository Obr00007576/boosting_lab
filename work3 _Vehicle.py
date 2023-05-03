from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

classes = ['van', 'saab', 'bus', 'opel']
Vehicle = pd.read_csv('./Vehicle.csv')
X = Vehicle.iloc[:, 1:19]
y = Vehicle.iloc[:, 19:20]

k = 6
model_num = 10

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()
#list of weights
weights = [np.full([X_train.shape[0]], 1, dtype = 'float32') for _ in range(model_num)]

def main():
    #boosting
    correct_rate = np.full(model_num, 0, dtype='float32')
    for boost in range(model_num):
        predicts = np.full([y_test.size], '', dtype=object)
        for x in range(y_test.size):
            distances = np.array(((X_train - X_test.iloc[x])**2).sum(axis=1), dtype=float)
            nn = np.argpartition(distances, k)[:k]
            classify = [0, 0, 0, 0]
            for i in range(len(classes)):
                for index in nn:
                    classify[i] += weights[boost][index] if classes[i] == y_train[index] else 0
            predict = classes[np.argmax(classify)]
            predicts[x] = predict
        error_index = np.where(np.not_equal(predicts, y_test))
        if boost < model_num-1:
            np.copyto(weights[boost+1], weights[boost])
            weights[boost+1][error_index] = 2 * weights[boost][error_index]
        print(np.sum(predicts != y_test)/y_test.size)
        correct_rate[boost] = np.sum(predicts == y_test)/y_test.size
    print(correct_rate)
    model_weights = correct_rate/correct_rate.sum()

    #testing
    all_predicts = []
    for boost in range(model_num):
        predicts = np.full([y_test.size], '', dtype=object)
        for x in range(y_test.size):
            distances = np.array(((X_train - X_test.iloc[x])**2).sum(axis=1), dtype=float)
            nn = np.argpartition(distances, k)[:k]
            classify = [0, 0, 0, 0]
            for i in range(len(classes)):
                for index in nn:
                    classify[i] += weights[boost][index] if classes[i] == y_train[index] else 0
            predict = classes[np.argmax(classify)]
            predicts[x] = predict
        all_predicts.append(predicts)

    
    boosting_data = pd.DataFrame(all_predicts).T
    boosting_pred = np.full([y_test.size], '', dtype=object)
    for i in range(y_test.size):
        pred = np.array(boosting_data.iloc[i])
        classify = [0, 0, 0, 0]
        for j in range(len(classes)):
            for m in range(model_num):
                classify[j] += model_weights[m] if classes[j] == pred[m] else 0
        boosting_pred[i] = classes[np.argmax(classify)]
    print("Boosting model error rate:")
    print(np.sum(boosting_pred != y_test)/y_test.size)

    




if __name__ == '__main__':
    main()
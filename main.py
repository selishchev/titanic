from titanic import *
import time
from multiprocessing import Process


def main():
    X, y = Data('train.csv').edit()
    t0 = time.time()
    p1 = Process(target=Processes().fit(X, y))
    p1.start()
    p1.join()
    print('Время выполнения процесса 1:', time.time() - t0, '\n')
    t0 = time.time()
    p2 = Process(target=Processes().process_two(X, y))
    p2.start()
    p2.join()
    print('Время выполнения процесса 2:', time.time() - t0, '\n')
    t0 = time.time()
    p3 = Process(target=Processes().process_three(X, y))
    p3.start()
    p3.join()
    print('Время выполнения процесса 3:', time.time() - t0)


if __name__ == '__main__':
    main()

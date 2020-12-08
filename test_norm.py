import numpy as np


def test_norm_equality(n_tests):

    count = 0
    for n in range(n_tests):

        size = np.random.randint(low=100,high=150)
        x = np.random.randn(size)

        a = np.linalg.norm(x,ord=4)**4
        b = np.outer(x,x.T)
        diag = np.diag(b)
        e = np.linalg.norm(diag,ord=2)**2
        square = x **2
        c = np.linalg.norm(b,ord="fro")**2
        d = np.trace(b.T @ b)
        print(a,c,d,e)
        equal =  np.abs(a-d) < 1e-5
        
        if equal:
            count+=1

    print("Passed {}/{} tests".format(count,n_tests))

def test_square_equality(n_tests):

    count = 0
    for n in range(n_tests):

        size = np.random.randint(low=100,high=1000)
        x = np.random.randn(size)

        a = np.diag(np.outer(x,x.T))
        b = x **2

        equal =  np.abs(a-b) < 1e-5
        
        if np.all(equal):
            count+=1

    print("Passed {}/{} tests".format(count,n_tests))


def test_frob(n_tests):

    count = 0
    for n in range(n_tests):

        size = np.random.randint(low=100,high=150)
        x = np.random.randn(size)

        a = np.dot(x.T,x)
        
        b = np.outer(x,x.T)
        c = b.T @ b
        d = np.trace(c)
        print(a,np.sqrt(d))
        equal = np.abs(a-d) < 1e-5

        if equal:
            count+=1

    print("Passed {}/{} tests".format(count,n_tests))

test_norm_equality(100)
#test_frob(10)
#test_square_equality(1000)

from Loader import Loader

if __name__=='__main__':

    loader = Loader(reduction=80)
    ctrain, ctest = loader.run()
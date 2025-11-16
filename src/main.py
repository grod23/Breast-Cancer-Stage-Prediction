from train import Train

def main():
    train = Train()
    print(train.model)
    train.train()
    # train.load_model()
    train.results()
    # train.display_batch()

if __name__ == '__main__':
    main()

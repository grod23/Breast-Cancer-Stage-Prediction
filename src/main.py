from train import Train

def main():

    train = Train()
    print(train.model)
    train.train()
    train.results()
    train.save_model()
    # train.display_batch()


if __name__ == '__main__':
    main()

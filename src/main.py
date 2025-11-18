from train import Train
from occlusion import Occlusion_Sensitivity

def main():
    train = Train()
    occlusion = Occlusion_Sensitivity(model=train.model, loader=train.testing_loader)
    # occlusion.plot_heatmap()
    print(train.model)
    train.train()
    # train.save_model()
    # train.load_model()
    train.results()
    # train.display_batch()

if __name__ == '__main__':
    main()

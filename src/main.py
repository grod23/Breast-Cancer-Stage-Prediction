from train import Train
from occlusion import Occlusion_Sensitivity

def main():
    train = Train()
    print(train.model)
    train.display_batch()
    occlusion = Occlusion_Sensitivity(model=train.model, loader=train.testing_loader)
    # occlusion.plot_heatmap()
    train.train()
    # train.save_model()
    # train.load_model()
    train.results()

if __name__ == '__main__':
    main()

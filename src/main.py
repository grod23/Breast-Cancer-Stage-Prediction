from train import Train
from gradcam import CustomGradCAM

def main():
    train = Train()
    print(train.model)
    for name, module in train.model.named_modules():
        print(name)
    # train.display_batch()
    train.train()
    # train.save_model()
    # train.load_model()
    train.results()
    grad_cam = CustomGradCAM(
        model=train.model,
        loader=train.training_loader,
        num_heatmaps=20,
        batch_size=train.batch_size,
        target_patient='Breast_MRI_028'
    )
    grad_cam.plot_heatmap()

if __name__ == '__main__':
    main()

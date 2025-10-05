import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F 
import torch 
import torchvision.utils as vutils
import wandb
import torch
from torchmetrics.classification import BinaryAccuracy
import torchmetrics
from torchmetrics.classification import MulticlassAccuracy, MulticlassAUROC
from torchvision.models import efficientnet_v2_m, EfficientNet_V2_M_Weights

from losses import * 

class MyEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.conv1 = nn.Conv2d(1, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, 3, 2, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()

        self.conv5 = nn.Conv2d(128, 8, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(8)
        self.relu5 = nn.ReLU()

        self.conv6 = nn.Conv2d(8, 4, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(4)
        self.relu6 = nn.ReLU()

    def forward(self, x):
        enc = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        enc = self.pool2(self.relu2(self.bn2(self.conv2(enc))))
        enc = self.pool3(self.relu3(self.bn3(self.conv3(enc))))
        enc = self.relu4(self.bn4(self.conv4(enc)))
        enc = self.relu5(self.bn5(self.conv5(enc)))
        enc = self.conv6(enc)
        return enc 
    
    
class MyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Decoder using ConvTranspose2d
        self.deconv0 = nn.ConvTranspose2d(4, 128, kernel_size=4, stride=2, padding=1)  # Upsample to 8x8
        self.dbn0 = nn.BatchNorm2d(128)
        self.drelu0 = nn.ReLU()

        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)  # Upsample to 16x16
        self.dbn1 = nn.BatchNorm2d(64)
        self.drelu1 = nn.ReLU()

        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # Upsample to 32x32
        self.dbn2 = nn.BatchNorm2d(32)
        self.drelu2 = nn.ReLU()

        self.deconv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)  # Upsample to 64x64
        self.dbn3 = nn.BatchNorm2d(16)
        self.drelu3 = nn.ReLU()

        self.deconv4 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)  # Upsample to 128x128
        self.dbn4 = nn.BatchNorm2d(8)
        self.drelu4 = nn.ReLU()

        # for the tumor head 
        self.deconv5 = nn.ConvTranspose2d(8, 1, kernel_size=4, stride=2, padding=1)  # Upsample to 256x256


        # for the mask head 
        self.deconv6 = nn.ConvTranspose2d(8, 8, kernel_size=4, stride=2, padding=1)  # Upsample to 256x256
        self.deconv7 = nn.ConvTranspose2d(8, 1, kernel_size=1, stride=1, padding=0)  # Upsample to 256x256

    def forward(self, enc):
        dec = self.drelu0(self.dbn0(self.deconv0(enc)))
        dec = self.drelu1(self.dbn1(self.deconv1(dec)))
        dec = self.drelu2(self.dbn2(self.deconv2(dec)))
        dec = self.drelu3(self.dbn3(self.deconv3(dec)))
        dec = self.drelu4(self.dbn4(self.deconv4(dec)))
        dec1 = self.deconv5(dec)  # Final layer, no activation to retain pixel values # tumor head 
        dec2 = self.deconv6(dec)     # mask head 
        dec2 = self.deconv7(dec2)  

        return dec1, dec2 

    

# Define the CNN Model using PyTorch Lightning
class Compress_Segmentor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encode = MyEncoder()
        self.decode = MyDecoder()
        self.hybrid_loss = HybridLoss()
        self.tverskyloss = TverskyLoss()

    def forward(self, x):
        x = self.encode(x)
        x1, x2 = self.decode(x)
        return x1, x2  #F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y_temp = batch
        y_tumor, y_mask = self(x)
        y = x*y_temp
        loss =  self.tverskyloss(y_mask, y_temp)  + torch.log(1+ F.mse_loss(y_tumor, y))
        #acc = (y_hat.argmax(dim=1) == y).float().mean()
        #psnr = 
        self.log("train/loss", loss, prog_bar=True)
        #self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_temp = batch
        y_tumor, y_mask = self(x)
        y = x*y_temp
        loss =  self.tverskyloss(y_mask, y_temp)  + torch.log(1+ F.mse_loss(y_tumor, y))
        #loss = torch.log(1+F.mse_loss(y_hat, y))
        #acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("validation/loss", loss, prog_bar=True)
        #self.log("val_acc", acc, prog_bar=True)

         # Log images (only log a few samples)
        if batch_idx == 0:
            if isinstance(self.logger, pl.loggers.WandbLogger) and batch_idx == 0:
                grid_input = vutils.make_grid(y[:8], normalize=True, scale_each=True)
                grid_pred = vutils.make_grid(y_tumor[:8], normalize=True, scale_each=True)

                self.logger.experiment.log({
                    "val/input_images": wandb.Image(grid_input, caption="Input Images"),
                    "val/predicted_images": wandb.Image(grid_pred, caption="Predicted Images"),
                    "global_step": self.trainer.global_step
                })

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    

    
class FCNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.3):
        super(FCNetwork, self).__init__()
        layers = []
        prev_size = input_size

        self.conv1 = nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0)  # Upsample to 256x256
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0)  # Upsample to 256x256
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(4, 4, kernel_size=1, stride=1, padding=0)  # Upsample to 256x256
        
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x)) 
        x = self.conv3(x)
        x = x.reshape(x.shape[0], -1)
        return self.model(x)

    

# this model is binary classification model: malignant, benign    
class BinaryClassification(pl.LightningModule):
    def __init__(self, input_dim=8192*2, num_classes = 1,  lr=1e-3, weight_decay=1e-5, encoder_weight_path = None):
        super().__init__()
        
        self.input_size = input_dim 
        self.hidden_sizes = [512, 128, 64, 32]
        self.output_size = num_classes

        segmodel = Compress_Segmentor.load_from_checkpoint(encoder_weight_path, strict=True)
        encoder = segmodel.encode 
        encoder.to(device = "cpu")
        self.encoder = encoder

        # **Freeze the encoder weights**
        for param in self.encoder.parameters():
            param.requires_grad = False


        self.linear = FCNetwork(input_size= self.input_size, hidden_sizes=self.hidden_sizes, output_size= self.output_size)

        self.loss_fn = nn.BCEWithLogitsLoss()  # More stable than BCELoss
        self.accuracy_metric = BinaryAccuracy()  # Accuracy metric using TorchMetrics
        self.auc_metric = torchmetrics.AUROC(task="binary")

        # Save hyperparameters
        self.save_hyperparameters()
        self.lr = lr
    
    def on_train_epoch_start(self):
        self.accuracy_metric.reset()
        self.auc_metric.reset()

    def on_validation_epoch_start(self):
        self.accuracy_metric.reset()
        self.auc_metric.reset()

    def on_test_epoch_start(self):
        self.accuracy_metric.reset()
        self.auc_metric.reset() 
        
    def _common_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(
            scores, y.float()
        )  # Ensure labels are float for BCEWithLogitsLoss
        return loss, scores, y, x 

    def forward(self, x):
        x = self.encoder(x)
        #x = x.reshape(x.shape[0], -1)
        return self.linear(x).squeeze()

    def training_step(self, batch, batch_idx):
        loss, scores, y, _ = self._common_step(batch, batch_idx)
        probs = torch.sigmoid(scores)  # Convert logits to probabilities

        # Update metrics
        self.accuracy_metric.update(probs, y.int())
        self.auc_metric.update(probs, y.int())

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y, x = self._common_step(batch, batch_idx)
        probs = torch.sigmoid(scores)

        # Update metrics
        self.accuracy_metric.update(probs, y.int())
        self.auc_metric.update(probs, y.int())

        self.log("validation/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if isinstance(self.logger, pl.loggers.WandbLogger) and batch_idx == 0:
                grid_input = vutils.make_grid(x[:8], normalize=True, scale_each=True)
                self.logger.experiment.log({
                    "validation/input_images": wandb.Image(grid_input, caption="Input Images"),
                    "global_step": self.trainer.global_step
                })

        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y, _  = self._common_step(batch, batch_idx)
        probs = torch.sigmoid(scores)

        # Update metrics
        self.accuracy_metric.update(probs, y.int())
        self.auc_metric.update(probs, y.int())

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train/accuracy", self.accuracy_metric.compute(), prog_bar=True)
        self.log("train/auc", self.auc_metric.compute(), prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("validation/accuracy", self.accuracy_metric.compute(), prog_bar=True)
        self.log("validation/auc", self.auc_metric.compute(), prog_bar=True)

    def on_test_epoch_end(self):
        self.log("test/accuracy", self.accuracy_metric.compute(), prog_bar=True)
        self.log("test/auc", self.auc_metric.compute(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    

# this model is multiclass classification model for mmotu dataset: malignant, benign    
class MultiClassClassification(pl.LightningModule):
    def __init__(self, input_dim=8192*2, num_classes = 8,  lr=1e-3, weight_decay=1e-5, encoder_weight_path = None):
        super().__init__()
        
        self.input_size = input_dim 
        self.hidden_sizes = [512, 128, 64, 32]
        self.output_size = num_classes
        self.num_classes = num_classes
        segmodel = Compress_Segmentor.load_from_checkpoint(encoder_weight_path, strict=True)
        encoder = segmodel.encode 
        encoder.to(device = "cpu")
        self.encoder = encoder

        # **Freeze the encoder weights**
        for param in self.encoder.parameters():
            param.requires_grad = False


        self.linear = FCNetwork(input_size= self.input_size, hidden_sizes=self.hidden_sizes, output_size= self.output_size)

        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy_metric = MulticlassAccuracy(num_classes=self.num_classes)
        self.auc_metric = MulticlassAUROC(num_classes=self.num_classes, average="macro")

        self.save_hyperparameters()
        self.lr = lr

    
    def on_train_epoch_start(self):
        self.accuracy_metric.reset()
        self.auc_metric.reset()

    def on_validation_epoch_start(self):
        self.accuracy_metric.reset()
        self.auc_metric.reset()

    def on_test_epoch_start(self):
        self.accuracy_metric.reset()
        self.auc_metric.reset() 
        
    def _common_step(self, batch, batch_idx):
        x, mask, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(
            scores, y.long()
        )  # Ensure labels are float for BCEWithLogitsLoss
        return loss, scores, y, x 

    def forward(self, x):
        x = self.encoder(x)
        #x = x.reshape(x.shape[0], -1)
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        loss, scores, y, _ = self._common_step(batch, batch_idx)
        probs = torch.softmax(scores, dim = 1)  # Convert logits to probabilities

        # Update metrics
        self.accuracy_metric.update(probs, y.int())
        self.auc_metric.update(probs, y.int())

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y, x = self._common_step(batch, batch_idx)
        probs = torch.softmax(scores, dim = 1)

        # Update metrics
        self.accuracy_metric.update(probs, y.int())
        self.auc_metric.update(probs, y.int())

        self.log("validation/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if isinstance(self.logger, pl.loggers.WandbLogger) and batch_idx == 0:
                grid_input = vutils.make_grid(x[:8], normalize=True, scale_each=True)
                self.logger.experiment.log({
                    "validation/input_images": wandb.Image(grid_input, caption="Input Images"),
                    "global_step": self.trainer.global_step
                })

        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y, _  = self._common_step(batch, batch_idx)
        probs = torch.softmax(scores)

        # Update metrics
        self.accuracy_metric.update(probs, y.int())
        self.auc_metric.update(probs, y.int())

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train/accuracy", self.accuracy_metric.compute(), prog_bar=True)
        self.log("train/auc", self.auc_metric.compute(), prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("validation/accuracy", self.accuracy_metric.compute(), prog_bar=True)
        self.log("validation/auc", self.auc_metric.compute(), prog_bar=True)

    def on_test_epoch_end(self):
        self.log("test/accuracy", self.accuracy_metric.compute(), prog_bar=True)
        self.log("test/auc", self.auc_metric.compute(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    



# this model is multiclass classification model for mmotu dataset: malignant, benign    
class MultiClassClassificationEfficientNet(pl.LightningModule):
    def __init__(self, num_classes = 8,  lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.output_size = num_classes
        self.num_classes = num_classes
        
        self.model = efficientnet_v2_m(num_classes = self.num_classes)

        # Modify the first conv layer to accept 1-channel input
        old_conv = self.model.features[0][0]
        self.model.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy_metric = MulticlassAccuracy(num_classes=self.num_classes)
        self.auc_metric = MulticlassAUROC(num_classes=self.num_classes, average="macro")

        self.save_hyperparameters()
        self.lr = lr
    
    def on_train_epoch_start(self):
        self.accuracy_metric.reset()
        self.auc_metric.reset()

    def on_validation_epoch_start(self):
        self.accuracy_metric.reset()
        self.auc_metric.reset()

    def on_test_epoch_start(self):
        self.accuracy_metric.reset()
        self.auc_metric.reset() 
        
    def _common_step(self, batch, batch_idx):
        x, mask, y = batch
        scores = self.forward(x)
        loss = self.loss_fn(
            scores, y.long()
        )  # Ensure labels are float for BCEWithLogitsLoss
        return loss, scores, y, x 

    def forward(self, x):
        x = self.model(x)
        return x 

    def training_step(self, batch, batch_idx):
        loss, scores, y, _ = self._common_step(batch, batch_idx)
        probs = torch.softmax(scores, dim = 1)  # Convert logits to probabilities

        # Update metrics
        self.accuracy_metric.update(probs, y.int())
        self.auc_metric.update(probs, y.int())

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, scores, y, x = self._common_step(batch, batch_idx)
        probs = torch.softmax(scores, dim = 1)

        # Update metrics
        self.accuracy_metric.update(probs, y.int())
        self.auc_metric.update(probs, y.int())

        self.log("validation/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        if isinstance(self.logger, pl.loggers.WandbLogger) and batch_idx == 0:
                grid_input = vutils.make_grid(x[:8], normalize=True, scale_each=True)
                self.logger.experiment.log({
                    "validation/input_images": wandb.Image(grid_input, caption="Input Images"),
                    "global_step": self.trainer.global_step
                })

        return loss

    def test_step(self, batch, batch_idx):
        loss, scores, y, _  = self._common_step(batch, batch_idx)
        probs = torch.softmax(scores)

        # Update metrics
        self.accuracy_metric.update(probs, y.int())
        self.auc_metric.update(probs, y.int())

        self.log("test/loss", loss, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train/accuracy", self.accuracy_metric.compute(), prog_bar=True)
        self.log("train/auc", self.auc_metric.compute(), prog_bar=True)

    def on_validation_epoch_end(self):
        self.log("validation/accuracy", self.accuracy_metric.compute(), prog_bar=True)
        self.log("validation/auc", self.auc_metric.compute(), prog_bar=True)

    def on_test_epoch_end(self):
        self.log("test/accuracy", self.accuracy_metric.compute(), prog_bar=True)
        self.log("test/auc", self.auc_metric.compute(), prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    

if __name__ == '__main__':
    # import torch 
    # model = ConvAutoEncoder()
    # model = BinaryClassification2(input_dim= 8192*2)
    # print(model)
    # model.eval()
    # print(model(torch.randn(1, 1,256, 256), torch.randn(1, 1,256, 256)).shape)

    model = MyEncoder()
    model.eval()
    #print(model)
    print(model(torch.randn(1, 1,256, 256)).shape)
    
    input_size = 64
    hidden_sizes = [512, 128, 64, 32]
    model5 = FCNetwork(input_size= 64, hidden_sizes= hidden_sizes, output_size= 8)
    print(model5(torch.randn(2, 4, 4, 4)).shape)



    model2 = MyDecoder()
    model2.eval()
    #print(model2)
    print(model2(torch.randn(1, 4, 4, 4))[0].shape)

    model3 = MultiClassClassificationEfficientNet(num_classes= 8)

    model4 = MultiClassClassification(input_dim= 64,num_classes= 8,  encoder_weight_path = r"checkpoints\normtverskyloss_binary_segmentation\0696540\best-checkpoint-epoch=92-validation\loss=0.2768.ckpt")

    print(model3(torch.randn(2, 1, 256, 256)).shape)
    print(model4(torch.randn(2, 1, 256, 256)).shape)

    
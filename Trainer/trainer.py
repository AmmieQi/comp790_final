import torch
import itertools
import numpy as np
import sys
sys.path.append('../')
import models

class Trainer(object):

    '''
    Trainer to train our method for disentangled representations
    '''

    def __init__(self, loss_weights, encoder, cls_head, fusion, style_discriminator,
            content_discriminator, device, save_dir, lr, verbose = False):

        self.loss_weights = loss_weights

        self.device = device

        self.style_encoder = encoder.to(self.device)
        self.content_encoder = encoder.to(self.device)
        self.cls_head = cls_head.to(self.device)
        self.fusion = fusion.to(self.device)
        self.style_discrim = style_discriminator.to(self.device)
        self.content_discrim = content_discriminator.to(self.device)

        self.verbose = verbose

        self.save_dir = save_dir

        self.cls_criterion = torch.nn.CrossEntropyLoss(ignore_index = -1)
        self.binary_criterion = torch.nn.BCEWithLogitsLoss()

        self.optim_S_D = torch.optim.SGD( self.style_discrim.parameters(), lr = lr, momentum = 0.9, nesterov = True )
        self.optim_C_D = torch.optim.SGD( self.content_discrim.parameters(), lr = lr, momentum = 0.9, nesterov = True )
        self.optim_CLS = torch.optim.SGD(itertools.chain(
                                    self.cls_head.parameters(),
                                    self.fusion.parameters(),
                                    self.style_encoder.parameters(),
                                    self.content_encoder.parameters()) ,
                                            lr = lr, momentum = 0.9, nesterov = True )


    def pass_input(self, data_input):

        """Takes input for training.

        Parameters
        ----------

        data_input:
          Data input for one iteration:
              mnist_data, mnist_label = next(mnist_train_cycle)
              svhn_data, svhn_label = next(svhn_train_cycle)
          Expects a list: [source_data, source_label, target_data, target_label]
        """
        source_data, source_label, target_data, target_label = data_input

        self.source_data = source_data.to(self.device)
        self.source_labels = source_label.to(self.device)
        self.target_data = target_data.to(self.device)
        self.target_labels = target_label.to(self.device)

        self.source_style_labels = torch.tensor(np.ones((self.source_data.size(0))), requires_grad = False).float().to(self.device)
        self.target_style_labels = torch.tensor(np.zeros((self.target_data.size(0))), requires_grad = False).float().to(self.device)

    def train(self):

        self.encode()
        self.update_weights()

    def encode(self):
        #Encode Style and Content
        self.source_style = self.style_encoder(self.source_data)
        self.target_style = self.style_encoder(self.target_data)
        self.source_content = self.content_encoder(self.source_data)
        self.target_content = self.content_encoder(self.target_data)


        self.sourceStyle_sourceContent = self.fusion(self.source_style, self.source_content, 0)
        self.sourceStyle_targetContent = self.fusion(self.source_style, self.target_content, 0)
        self.targetStyle_sourceContent = self.fusion(self.target_style, self.source_content, 1)
        self.targetStyle_targetContent = self.fusion(self.target_style, self.target_content, 1)

    def backward_Style_G(self):

        """
        Calculate loss for content encoder
        """

        #Source
        pred_source = self.style_discrim(self.source_content)
        loss_G_source = self.binary_criterion(pred_source, self.target_style_labels)
        # Target
        pred_target = self.style_discrim(self.target_content)
        loss_G_target = self.binary_criterion(pred_target, self.source_style_labels)

        # Combined loss and calculate gradients
        loss_G = (loss_G_source + loss_G_target) * 0.5
        loss_G *= self.loss_weights.style_generator_loss
        loss_G.backward(retain_graph = True)

        return loss_G



    def backward_Style_D(self):

        """
        Calculate gan loss for style discriminator
            The input to the style discriminator is the content encodings
        """

        # Source
        pred_source = self.style_discrim(self.source_content.detach())
        loss_D_source = self.binary_criterion(pred_source, self.source_style_labels)
        # Target
        pred_target = self.style_discrim(self.target_content.detach())
        loss_D_target = self.binary_criterion(pred_target, self.target_style_labels)

        # Combined loss and calculate gradients
        loss_D = (loss_D_source + loss_D_target) * 0.5
        loss_D *= self.loss_weights.style_discriminator_loss
        loss_D.backward()

        return loss_D


    def backward_Content_G(self):
        '''
        Calculate loss for style encoder
        '''

        #Source
        pred_source = self.content_discrim(self.source_style )
        loss_G_source = self.get_entropy_loss(pred_source)
        #Target
        pred_target = self.content_discrim(self.target_style )
        loss_G_target = self.get_entropy_loss(pred_target)

        loss_G = (loss_G_source + loss_G_target) * 0.5
        loss_G *= self.loss_weights.content_generator_loss
        loss_G.backward(retain_graph = True)

        return loss_G


    def backward_Content_D(self):

        """

        """
        #Source
        pred_source = self.content_discrim(self.source_style.detach())
        loss_D_source = self.cls_criterion(pred_source, self.source_labels)
        #Target
        pred_target = self.content_discrim(self.target_style.detach())
        loss_D_target = self.cls_criterion(pred_target, self.target_labels)

        loss_D = (loss_D_source + loss_D_target) * 0.5
        loss_D *= self.loss_weights.content_discriminator_loss
        loss_D.backward()

        return loss_D

    def backward_CLS(self):

        '''
        Computes loss for classifier head
        '''

        self.sourceStyle_sourceContent_preds = self.cls_head(self.sourceStyle_sourceContent)
        self.sourceStyle_targetContent_preds = self.cls_head(self.sourceStyle_targetContent)
        self.targetStyle_sourceContent_preds = self.cls_head(self.targetStyle_sourceContent)
        self.targetStyle_targetContent_preds = self.cls_head(self.targetStyle_targetContent)

        sourceStyle_sourceContent_preds = self.cls_criterion(self.sourceStyle_sourceContent_preds, self.source_labels)
        sourceStyle_targetContent_preds = self.cls_criterion(self.sourceStyle_targetContent_preds, self.target_labels)
        targetStyle_sourceContent_preds = self.cls_criterion(self.targetStyle_sourceContent_preds, self.source_labels)
        targetStyle_targetContent_preds = self.cls_criterion(self.targetStyle_targetContent_preds, self.target_labels)

        cls_loss = (sourceStyle_sourceContent_preds + sourceStyle_targetContent_preds + targetStyle_sourceContent_preds + targetStyle_targetContent_preds) / 4.
        # cls_loss = (sourceStyle_sourceContent_preds + targetStyle_sourceContent_preds ) / 2.
        cls_loss *= self.loss_weights.cls_loss
        cls_loss.backward()

        return cls_loss



    def update_weights(self):

        self.optim_S_D.zero_grad()
        self.backward_Style_D()
        self.optim_S_D.step()

        self.optim_C_D.zero_grad()
        self.backward_Content_D()
        self.optim_C_D.step()

        self.optim_CLS.zero_grad()
        self.backward_Style_G()
        self.backward_Content_G()
        self.backward_CLS()
        self.optim_CLS.step()


    def load_model(self):
        pass

    def save_model(self):
        pass

    def get_entropy_loss(self, out):
        return -torch.mean(torch.log(torch.nn.functional.softmax(out + 1e-6, dim=-1)))

"""    The function bbox_loss calculates the loss and accuracy of the bbox (don't think that's true, it's not used)"""import torchimport torch.nn as nnimport torch.nn.functional as Fimport randomimport torch.optim as optimimport timeimport datetimeprint('chip')class BBoxTracker(nn.Module):    def __init__(self,n=32,size_big=(200,64),bbox_dist=(8,8)):        super(BBoxTracker,self).__init__()        self.nr = bbox_dist[0]        self.nv = bbox_dist[1]        self.L = self.nr*self.nv*4        self.Nr = size_big[0]        self.Nv = size_big[1]        self.norm1 = nn.BatchNorm2d(1)        self.MBC1 = MovingBiasConv(in_channels=1,out_channels=n,Nr=self.Nr,Nv=self.Nv)        self.MBC2 = MovingBiasConv(in_channels=n,out_channels=n*2,Nr=self.Nr,Nv=self.Nv)        self.MBC3 = MovingBiasConv(in_channels=n*2,out_channels=n*4,Nr=self.Nr,Nv=self.Nv)        self.MBC4 = MovingBiasConv(in_channels=n*4,out_channels=n*4,Nr=self.Nr,Nv=self.Nv)        self.block = nn.Sequential(            nn.Conv2d(in_channels=4 * n, out_channels=2 * n, kernel_size=3, padding=1),            nn.Dropout(0.25),            nn.LeakyReLU(),            nn.Conv2d(in_channels=2 * n, out_channels=2 * n, kernel_size=3, padding=1),            nn.Dropout(0.25),            nn.LeakyReLU(),            nn.Conv2d(in_channels=2 * n, out_channels= n, kernel_size=3, padding=1),            nn.Dropout(0.25),            nn.LeakyReLU(),            nn.Conv2d(in_channels=n,out_channels=1,kernel_size=1,padding=0)        )    def forward(self,x,center=None,restore=True,shuffle=False):        batch_size = x.shape[0]#        x = self.norm1(x)        if center is None:            bbox = None        else:            bbox = self.get_bbox(center,shuffle=shuffle)            x = self.crop(x,bbox)        shape_crop = x.shape        x = self.MBC1(x,bbox)        x = self.MBC2(x,bbox)        x = self.MBC3(x,bbox)        x = self.MBC4(x,bbox)        x = self.block(x)        x = x.view(batch_size, -1)        x = F.log_softmax(x,dim=1)        if restore:            x = x.view(shape_crop)        return x,bbox    def crop(self,x,bboxs):        batch_size = x.shape[0]        c = x.shape[1]        y = torch.zeros([batch_size, c, self.nr*2, self.nv*2])        for i in range(batch_size):            bbox = bboxs[i]            y[i, :, :, :] = x[i, :, bbox[0]:bbox[2], bbox[1]:bbox[3]]        return y    def get_bbox(self, center, shuffle=False):        batch_size = center.shape[0]        bboxs = list()        for i in range(batch_size):            r_idx = center[i, 0]            v_idx = center[i, 1]            if shuffle:                r_idx += random.randint(-self.nr+1, self.nr)                v_idx += random.randint(-self.nv+1, self.nv)            r = max(self.nr, min(self.Nr-self.nr, r_idx))            v = max(self.nv, min(self.Nv-self.nv, v_idx))            r_min = int(r - self.nr)            r_max = int(r + self.nr)            v_min = int(v - self.nv)            v_max = int(v + self.nv)            bbox = [r_min, v_min, r_max, v_max]            bboxs.append(bbox)        return bboxsclass MovingBiasConv(nn.Module):    def __init__(self,in_channels,out_channels,Nr,Nv):        super(MovingBiasConv,self).__init__()#        self.pad = nn.ConstantPad2d(1,0)        self.bias = nn.Parameter(torch.randn(in_channels,Nr,Nv))        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,bias=True)        self.drop = nn.Dropout(0.25)        self.LeakyRelu = nn.LeakyReLU()    def forward(self,x,bboxs=None):        batch_size = x.shape[0]        c = x.shape[1]        nr = x.shape[2]        nv = x.shape[3]        if bboxs is None:            if c != self.bias.shape[0] or nr != self.bias.shape[1] or nv != self.bias.shape[2]:                raise ValueError(f"input of shape {x.shape} does not match with bias tensor of shape {self.bias.shape} "                                 f"Please make sure that the setup of the network matches the input or use a Bounding "                                 f"Box")            x = x - self.bias        else:            for i in range(batch_size):                bbox = bboxs[i]                x[i,:,:,:] = x[i,:,:,:] - self.bias[:, bbox[0]:(bbox[0]+nr), bbox[1]:(bbox[1]+nv)]        x = self.conv(x)        x = self.drop(x)        x = self.LeakyRelu(x)        return xdef bbox_loss(model, input, labels=None, lambd=0.0, bbox=True):    batch_size = input.shape[0]    one_hot = torch.zeros(input.shape)    for i in range(batch_size):        label = labels[i]        r = label[0]        v = label[1]        one_hot[i, 0, r, v] = 1    one_hot = one_hot.detach()    if bbox:        x, bboxs = model(input, labels, restore=False, shuffle=True)        y = model.crop(one_hot, bboxs).view(batch_size, -1)        w = 255 * y.detach() + 255    else:        x, bboxs = model(input, restore=False, shuffle=True)        y = one_hot.view(batch_size, -1)        w = 12799 * y.detach() + 12799    estim = torch.argmax(x, dim=1).long()    GT = torch.argmax(y, dim=1).long().detach()    acc = 0.0    for i in range(batch_size):        if estim[i] == GT[i]:            acc += 1    acc /= batch_size    loss_BCE = F.binary_cross_entropy(torch.exp(x), y, weight=w)    if lambd > 0:        loss_CE = F.cross_entropy(x, GT)        loss = loss_CE * lambd + loss_BCE * (1 - lambd)    else:        loss = loss_BCE    return loss, accdef format_time(seconds):    return str(datetime.timedelta(seconds=int(seconds)))def train_bbox_and_frame(model,                         train_loader,                         valid_loader,                         learning_rate=0.0001,                         epochs=100,                         epochs_0=50,                         weight_decay: float = 0,                         checkpoint_path=None):    train_loss_list = list()    train_acc_list = list()    valid_loss_list = list()    valid_acc_list = list()    optimizer = optim.Adam(model.parameters(),                           lr=learning_rate,                           weight_decay=weight_decay,                           betas=(0.90, 0.999))    for epoch in range(epochs):        start_time = time.time()        train_loss = 0.0        train_acc_bbox = 0.0        train_acc_frame = 0.0        valid_loss = 0.0        valid_acc_bbox = 0.0        valid_acc_frame = 0.0        if epoch < epochs_0:            lambd = 0.3        else:            lambd = 0.0        # Train        model.train()        for i, (input, label) in enumerate(train_loader):            torch.cuda.empty_cache()            optimizer.zero_grad()            loss_bbox, acc_bbox = bbox_loss(model,input.detach().clone(),label.detach().clone(),lambd)            loss_frame, acc_frame = bbox_loss(model, input.detach().clone(),label.detach().clone(), bbox=False)            loss = loss_bbox + loss_frame/2            loss.backward()            optimizer.step()            train_loss += loss.item()            train_acc_bbox += acc_bbox            train_acc_frame += acc_frame        model.eval()        # Validate        with torch.no_grad():            for i, (input, label) in enumerate(valid_loader):                torch.cuda.empty_cache()                loss_bbox, acc_bbox = bbox_loss(model, input.detach().clone(), label.detach().clone())                loss_frame, acc_frame = bbox_loss(model, input.detach().clone(), label.detach().clone(),bbox=False)                loss = loss_bbox + loss_frame/50                valid_loss += loss.item()                valid_acc_bbox += acc_bbox                valid_acc_frame += acc_frame        # Print statistics        train_loss /= len(train_loader)        train_acc_bbox /= len(train_loader)        train_acc_frame /= len(train_loader)        valid_loss /= len(valid_loader)        valid_acc_bbox /= len(valid_loader)        valid_acc_frame /= len(valid_loader)        train_loss_list.append(train_loss)        train_acc_list.append(train_acc_bbox)        valid_loss_list.append(valid_loss)        valid_acc_list.append(valid_acc_bbox)        epoch_time = time.time() - start_time        remaining_time = epoch_time * (epochs - epoch - 1)        formatted_epoch_time = format_time(epoch_time)        formatted_remaining_time = format_time(remaining_time)        print(f"Epoch {epoch + 1}/{epochs}")        print(f"Train Loss: {train_loss:.4f},  Valid Loss: {valid_loss:.4f}")        print(f"Train BBox Acc: {train_acc_bbox * 100:.2f}%, Valid BBox Acc: {valid_acc_bbox * 100:.2f}%")        print(f"Train Frame Acc: {train_acc_frame * 100:.2f}%, Valid Frame Acc: {valid_acc_frame * 100:.2f}%")        print(f"Time taken: {formatted_epoch_time}, Estimated remaining time: {formatted_remaining_time}")        if checkpoint_path is not None:            torch.save(model.state_dict(), checkpoint_path)    return train_loss_list, train_acc_list, valid_loss_list, valid_acc_listdef train_bbox(model,               train_loader,               valid_loader,               learning_rate=0.0001,               epochs=100,               epochs_0=50,               weight_decay: float = 0,               checkpoint_path=None):    train_loss_list = list()    train_acc_list = list()    valid_loss_list = list()    valid_acc_list = list()    optimizer = optim.Adam(model.parameters(),                           lr=learning_rate,                           weight_decay=weight_decay,                           betas=(0.90, 0.999))    val_loss_best = torch.tensor(float('inf'))    for epoch in range(epochs):        start_time = time.time()        train_loss = 0.0        train_acc_bbox = 0.0        valid_loss = 0.0        valid_acc_bbox = 0.0        if epoch < epochs_0:            lambd = 0.3        else:            lambd = 0.0        # Train        model.train()        for i, (input, label) in enumerate(train_loader):            torch.cuda.empty_cache()            optimizer.zero_grad()            loss_bbox, acc_bbox = bbox_loss(model, input.detach(), label.detach(),lambd)            loss = loss_bbox            loss.backward()            optimizer.step()            train_loss += loss.item()            train_acc_bbox += acc_bbox        model.eval()        # Validate        with torch.no_grad():            for i, (input, label) in enumerate(valid_loader):                torch.cuda.empty_cache()                loss_bbox, acc_bbox = bbox_loss(model, input.detach(), label.detach())                loss = loss_bbox                valid_loss += loss.item()                valid_acc_bbox += acc_bbox        # Print statistics        train_loss /= len(train_loader)        train_acc_bbox /= len(train_loader)        valid_loss /= len(valid_loader)        valid_acc_bbox /= len(valid_loader)        train_loss_list.append(train_loss)        train_acc_list.append(train_acc_bbox)        valid_loss_list.append(valid_loss)        valid_acc_list.append(valid_acc_bbox)        epoch_time = time.time() - start_time        remaining_time = epoch_time * (epochs - epoch - 1)        formatted_epoch_time = format_time(epoch_time)        formatted_remaining_time = format_time(remaining_time)        print(f"Epoch {epoch + 1}/{epochs}")        print(f"Train Loss: {train_loss:.4f},  Valid Loss: {valid_loss:.4f}")        print(f"Train BBox Acc: {train_acc_bbox * 100:.2f}%, Valid BBox Acc: {valid_acc_bbox * 100:.2f}%")        print(f"Time taken: {formatted_epoch_time}, Estimated remaining time: {formatted_remaining_time}")        if checkpoint_path is not None:            if valid_loss < val_loss_best:                val_loss_best = valid_loss                torch.save(model.state_dict(), checkpoint_path)    return train_loss_list, train_acc_list, valid_loss_list, valid_acc_list
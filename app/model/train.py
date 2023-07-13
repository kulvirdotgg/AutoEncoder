import torch
from model import Autoencoder, Encoder, Decoder, transform_img, add_noise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder()
model = model.to(device)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

epochs = 10

train_loss = 0.
recon = []

model.train()
for epoch in range(epochs):
    num_batches = 0
    loop = tqdm(train_dataloader)

    for image_batch, _ in loop:
        image_batch = image_batch.to(device)
        image_batch += torch.randn_like(image_batch)
        output = model(image_batch)
        loss = loss_fn(output, image_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        num_batches += 1
        loop.set_postfix(loss=loss.item())
    train_loss /= num_batches
    recon.append((image_batch, output))
    print(f'Epoch:{epoch+1}, Loss:{train_loss:.4f}')

torch.save(model.state_dict(), 'mnistae-0.1.0.pth')

model.eval()
model.load_state_dict(torch.load('mnistae-0.1.0.pth'))
test_loss, num_batches = 0., 0.
for image_batch, _ in test_dataloader:
    image_batch += torch.randn_like(image_batch)
    with torch.no_grad():
        image_batch = image_batch.to(device)
        output = model(image_batch)
        loss = loss_fn(output, image_batch)

        test_loss += loss.item()
        num_batches += 1
test_loss /= num_batches
print('Reconstruction error: %f' % (test_loss))

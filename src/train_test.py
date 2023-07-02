import torch as th
import torch.nn as nn

# training function


def train(
    epoch,
    train_loader,
    model,
    device,
    criterion,
    optimizer,
    train_losses,
    train_accu,
):
    print("\nEpoch : %d" % epoch)

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    accu = 100.0 * correct / total

    train_accu.append(accu)
    train_losses.append(train_loss)
    print("Train Loss: %.3f | Accuracy: %.3f" % (train_loss, accu))


##############################################################
# testing function


def test(test_loader, model, device, criterion, eval_losses, eval_accu):
    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with th.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_loss = running_loss / len(test_loader)
    accu = 100.0 * correct / total

    eval_losses.append(test_loss)
    eval_accu.append(accu)

    print("Test Loss: %.3f | Accuracy: %.3f" % (test_loss, accu))


###############################################################


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

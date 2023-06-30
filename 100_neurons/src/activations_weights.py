import numpy as np
import datetime
import torch as th


def get_activations(test_loader, device, model, save=False):
    output_activations = []  # List to store the activations
    predicted = []  # List to store the predicted label for each test data processed
    hidden_activations = []  # List to store the hidden layer activations

    def hook(module, input, output):
        hidden_activations.append(output.detach().cpu().numpy())

    # Register the hook to the desired hidden layer
    target_layer = model.fc1

    hook_handle = target_layer.register_forward_hook(hook)

    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        # Save the activations
        output_activations.append(output.detach().cpu().numpy())

        predicted.append((output.max(1)[1]))

    # Convert the list of activations into a single NumPy array
    hidden_activations = np.concatenate(hidden_activations, axis=0)
    hook_handle.remove()
    output_activations = np.concatenate(output_activations, axis=0)
    predicted = np.concatenate(predicted, axis=0)

    if save == True:
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y_%d_%m_%H_%M")

        hidden_activations_torch = th.from_numpy(hidden_activations)
        output_activations_torch = th.from_numpy(output_activations)
        predicted_torch = th.from_numpy(predicted)

        th.save(
            hidden_activations_torch,
            f"activations/hidde_activations_{formatted_time}.pt",
        )
        th.save(
            output_activations_torch,
            f"activations/output_activation{formatted_time}.pt",
        )

        th.save(
            predicted_torch,
            f"predicted/predicted_labels{formatted_time}.pt",
        )

    return hidden_activations, output_activations, predicted


def get_weights(model, save=False):
    weights_first_layer = model.fc1.weight.detach().cpu()
    weights_second_layer = model.fc2.weight.detach().cpu()

    if save == True:
        current_time = datetime.datetime.now()
        formatted_time = current_time.strftime("%Y_%d_%m_%H_%M")

        th.save(
            weights_first_layer,
            f"weights/weights_first_layer_{formatted_time}.pt",
        )
        th.save(
            weights_second_layer,
            f"weights/weights_second_layer_{formatted_time}.pt",
        )

    return weights_first_layer.numpy(), weights_second_layer.numpy()

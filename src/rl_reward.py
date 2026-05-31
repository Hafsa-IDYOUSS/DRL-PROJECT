import torch
import torch.nn.functional as F


def final_confidence_reward(
    model,
    initial_tensor,
    final_tensor,
    label,
    device,
    num_actions,
    action_penalty=0.01
):
    model.eval()

    initial_tensor = initial_tensor.unsqueeze(0).to(device)
    final_tensor = final_tensor.unsqueeze(0).to(device)
    label = torch.tensor([label]).to(device)

    with torch.no_grad():
        initial_output = model(initial_tensor)
        final_output = model(final_tensor)

        initial_probs = F.softmax(initial_output, dim=1)
        final_probs = F.softmax(final_output, dim=1)

        p_initial = initial_probs[0, label.item()].item()
        p_final = final_probs[0, label.item()].item()

    reward = (p_final - p_initial) - (action_penalty * num_actions)

    return reward, p_initial, p_final
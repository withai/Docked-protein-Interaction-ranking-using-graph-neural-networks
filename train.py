import torch
import torch.nn.functional as F

def train(model, device, train_loader, optimizer, epoch, mini_batches_per_epoch, two_graph_class_names, logger=None):
    model.train()
    mini_batch_count = 0
    epoch_loss = 0

    for batch_idx, local_batch in enumerate(train_loader):
        target = []
        mini_batch_output = []

        # try:
        for item in local_batch:
            if(item["vertices"].size()[0] == 0):
                if(logger is not None):
                    logger.error(item["name"])
                else:
                    print("Error: " + item["name"])
                continue

            # Move graph to GPU.
            vertices = item["vertices"].to(device)
            nh_indices = item["nh_indices"].to(device)
            int_indices = item["int_indices"].to(device)
            nh_edges = item["nh_edges"].to(device)
            int_edges = item["int_edges"].to(device)
            is_int = item["is_int"].to(device)
            scores = item["dockq_score"].to(device)

            target.append(scores)

            model_input = None
            if(model.conv1.__class__.__name__ in two_graph_class_names):
                model_input = (vertices, vertices, nh_indices, int_indices, nh_edges, int_edges, is_int)
            else:
                model_input = (vertices, nh_indices, int_indices, nh_edges, int_edges, is_int)

            output = model(model_input)

            mini_batch_output.append(output)

        output = torch.stack(mini_batch_output)
        if(model.multi_label):
            output = output.squeeze(1)
            target = torch.stack(target).view(-1, 5)
        else:
            target = torch.stack(target).view(-1, 1)

        # print("##########################")
        # print(output)
        # print(target)
        # print("##########################")

        optimizer.zero_grad()

        loss = model.loss(output, target, reduction='mean')
        loss.backward()
        optimizer.step()

        mini_batch_count += 1
        epoch_loss += loss.item()

        # except:
        #     print("Missed mini-batch")
        #     pass

        if(mini_batch_count > mini_batches_per_epoch):
            break

    model.scheduler.step()

    if(logger is not None):
        logger.info("Epoch " + str(epoch) + " Loss: " + str(epoch_loss/mini_batch_count))
    else:
        print("Epoch " + str(epoch) + " Loss: " + str(epoch_loss/mini_batch_count))

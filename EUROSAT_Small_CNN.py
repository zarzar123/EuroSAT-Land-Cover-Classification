import time

start_time = time.perf_counter()

with torch.inference_mode():

    for batch_index, (images, labels) in enumerate(test_loader):

        images = images.to(device)

        small_probs = torch.softmax(
            small(images), dim=1
        )

        resnet_probs = torch.softmax(
            resnet(images), dim=1
        )

        # Synchronize MPS before measuring elapsed time.
        if device.type == "mps":
            torch.mps.synchronize()

        if (batch_index + 1) % 10 == 0:

            elapsed = time.perf_counter() - start_time

            print(
                f"Batch {batch_index + 1}/{len(test_loader)} | "
                f"Elapsed: {elapsed:.1f}s",
                flush=True
            )
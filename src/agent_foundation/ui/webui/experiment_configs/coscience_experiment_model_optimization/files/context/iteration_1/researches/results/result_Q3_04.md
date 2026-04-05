Backward Pass, Bug Detection & torch.compile Compatibility
Graph Breaks from torch.no_grad() in a Compiled Forward Pass

Using torch.no_grad() inside a model’s forward will force a graph break under torch.compile. TorchDynamo (the graph capture in PyTorch 2.x) cannot trace through a no_grad region, so it splits the computation graph at that point. In practice, each use of no_grad in the forward pass yields a separate compiled segment. For example, decorating a function with @torch.no_grad() means the compiler breaks the graph when calling that function – the code before and after run in different graph segments. This incurs extra overhead because the compiler must end one fused kernel graph, execute the no-grad part eagerly, then resume a new graph. In a “typical” forward pass, every invocation of a no_grad context causes one graph break (splitting the graph into two parts around that region). Multiple no_grad blocks would lead to multiple breaks and thus multiple kernel launches. The key takeaway: even a single @torch.no_grad decorator will break the optimized graph once per use, hurting fusion and performance. It also has the critical effect of disabling gradient tracking through that entire function, which was the root bug here (no gradients flowing back at all).
torch.inference_mode() vs torch.no_grad() for Index Computation

Replacing torch.no_grad with torch.inference_mode is not recommended in this scenario. Both contexts disable autograd, and both will cause graph breaks if used inside a compiled model (neither will be traced through). In fact, early PyTorch 2.x reports showed that torch.inference_mode() could also trigger errors or slowdowns when used with torch.compile. Functionally, inference_mode is a more extreme version of no-grad (it avoids even tracking version counters for mutations), which yields speedups in pure eager inference. However, under torch.compile the difference is negligible – the compiler already minimizes Python overhead, so inference_mode doesn’t add much benefit. PyTorch developers note that in the “end state” of compile, no_grad and inference_mode should behave equivalently in terms of performance. Given that inference_mode historically had compatibility issues with torch.compile and provides no gradient tracking just like no-grad, there is little reason to switch. The safer approach is to continue using torch.no_grad (in a limited scope as discussed below) for computing indices. In summary: keep using no-grad for the index calculation – switching to inference mode won’t fix the graph break or gradient flow, and may introduce other complications, without any real upside in a training context.
Correct Pattern for Non-Differentiable Indexing Operations

When you have to perform indexing or other non-differentiable operations without allowing them to wreck your gradients, the pattern is to isolate the nondifferentiable part in a no-grad context, but leave the actual tensor indexing in the grad-enabled context. In other words, do not wrap the entire function in @no_grad. Instead, use a with torch.no_grad(): block only around the index computation (or any discrete logic), then use the resulting indices to index the tensor outside that block. For example:

with torch.no_grad():
    idx = compute_indices(...)   # e.g. argmax or other index logic (no grad needed)
# Autograd re-enabled here
output = seq_embeddings[idx]     # indexing uses idx, gradients flow into seq_embeddings

This way, the index tensor idx is computed without tracking (since its values are non-differentiable integers), but the actual selection seq_embeddings[idx] is executed with gradient tracking on. Autograd will correctly propagate gradients back to the seq_embeddings entries that were selected, while ignoring idx (which has no gradient). The bug in the original code was that wrapping the whole function in no-grad prevented any gradient from flowing through the indexing operation. The correct pattern above fixes that: it only stops grad where appropriate (computing indices), and allows the backward pass to see the indexing operation. Essentially, treat index selection as a differentiable gather on seq_embeddings – just ensure the indices are obtained in a grad-free manner (since index positions are inherently not differentiable in value). By scoping torch.no_grad tightly, you avoid unnecessary graph breaks and let gradients flow through the rest of the function.
Detecting Silent Gradient-Flow Bugs in Large Models

Silent gradient flow issues – where parts of the network stop receiving gradients without throwing errors – can be tricky to catch. Here are some strategies to detect them in large models:

    Gradient inspection for parameters: After running a training step (forward + backward), iterate over model parameters and check their .grad. If a parameter that should be learning consistently has grad=None (or a zero gradient) every time, that’s a red flag. In practice, you can log any parameter names with no gradients. For example, one debugging approach collects all parameters with p.grad is None into a list for review. This helped reveal that the encoder’s parameters never got grads due to the no-grad bug.

    Use hooks to track gradients: PyTorch’s hook system lets you monitor the backward pass. You can attach a backward hook to tensors or modules to verify that gradients propagate. For instance, registering a hook on an intermediate tensor (tensor.register_hook) will call your function with the grad for that tensor during backprop. Similarly, nn.Module.register_full_backward_hook can log gradient info for module outputs. If a hook on a certain layer’s output never fires or always receives None/zero, you’ve found a broken gradient path. In large networks, you might instrument critical layers (like the output of an encoder) with hooks that print or assert on gradient magnitude. PyTorch’s official tutorials demonstrate attaching hooks to all layers to collect their gradient values for analysis – this kind of visibility can pinpoint where grads vanish.

    Grad-check smaller subgraphs: If feasible, isolate the sub-module you suspect and use torch.autograd.grad or torch.autograd.gradcheck with a test input to ensure it produces gradients. This isn’t always practical at scale, but for a smaller component (like a custom operation), it can catch issues. For example, if you had a custom indexing routine, you could compare its output gradients to a reference implementation.

    Autograd anomaly detection: Enable anomaly detection with torch.autograd.set_detect_anomaly(True) during debugging runs. This mode will throw a detailed traceback whenever an operation in the backward pass fails to compute a gradient or encounters an invalid gradient state. It’s mainly meant for catching things like inappropriate in-place modifications or undefined gradients. While it won’t directly flag a deliberately disabled gradient (like a no-grad block), it will alert you if “an operation failed to compute its gradient” with a hint to the problematic op. In our context, if some part of the graph was unexpectedly cut off, anomaly detection might not always trigger, but it’s useful if any autograd error is the cause of missing grad.

    Gradient flow sanity checks and debugging mode: In a complex training loop, it can help to run a few iterations in a debug mode: disable optimizations like torch.compile or gradient checkpointing and see if gradients appear. Gradient checkpointing, in particular, complicates the backward graph (since it re-computes layers on the fly with no_grad in forward passes). If you suspect checkpointing might be hiding a bug, try turning it off temporarily – if gradients start flowing when checkpoints are disabled, it suggests the issue lies in that mechanism or how it’s used. In general, ensure that any use of torch.no_grad() (e.g. for evaluation or within checkpoint routines) isn’t inadvertently affecting your training graph.

    Logging and unit tests for gradients: It may sound obvious, but adding explicit logging of gradient norms for various parts of the model during training can quickly surface anomalies. For example, you can log the norm of gradients for the embedding layer or the encoder each iteration. If you see a constant zero norm where there shouldn’t be one, you know there’s trouble. Some practitioners even write small unit tests for gradient flow: e.g., do a forward/backward on a tiny example and assert that a particular submodule’s parameter grad is non-zero. This kind of “gradient unit test” can catch a silent break early in development.

By using these techniques – gradient hooks, direct inspections, anomaly detection, and careful ablation of features like checkpointing – you can systematically identify where the gradient flow stops. In this case, such methods would have pointed straight to the get_nro_embeddings function: its outputs’ gradient never reached the input embeddings, revealing the torch.no_grad decorator as the culprit. Once identified, the fix was to apply the pattern discussed above so that gradients can once again propagate as expected.
Citations

Torch.compile interaction with autocast and no_grad - torch.compile - PyTorch Forums
https://discuss.pytorch.org/t/torch-compile-interaction-with-autocast-and-no-grad/178648

torch.compile Troubleshooting — PyTorch 2.10 documentation
https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_troubleshooting.html

`torch.compile` + `torch.no_grad` not working for Mask R-CNN · Issue #97340 · pytorch/pytorch · GitHub
https://github.com/pytorch/pytorch/issues/97340

Performance of `torch.compile` is significantly slowed down under `torch.inference_mode` - torch.compile - PyTorch Forums
https://discuss.pytorch.org/t/performance-of-torch-compile-is-significantly-slowed-down-under-torch-inference-mode/191939

Performance of `torch.compile` is significantly slowed down under `torch.inference_mode` - torch.compile - PyTorch Forums
https://discuss.pytorch.org/t/performance-of-torch-compile-is-significantly-slowed-down-under-torch-inference-mode/191939

Are all operations defined in torch.Tensor differentiable? - PyTorch Forums
https://discuss.pytorch.org/t/are-all-operations-defined-in-torch-tensor-differentiable/4318

12 PyTorch Debugging Techniques That Catch Silent Bugs - Medium
https://medium.com/@Nexumo_/12-pytorch-debugging-techniques-that-catch-silent-bugs-9cfa11eaa360

Visualizing Gradients — PyTorch Tutorials 2.10.0+cu128 documentation
https://docs.pytorch.org/tutorials/intermediate/visualizing_gradients_tutorial.html

Visualizing Gradients — PyTorch Tutorials 2.10.0+cu128 documentation
https://docs.pytorch.org/tutorials/intermediate/visualizing_gradients_tutorial.html

Visualizing Gradients — PyTorch Tutorials 2.10.0+cu128 documentation
https://docs.pytorch.org/tutorials/intermediate/visualizing_gradients_tutorial.html

python - Explanation behind the following Pytorch results - Stack Overflow
https://stackoverflow.com/questions/62496172/explanation-behind-the-following-pytorch-results

Understand Gradient Checkpoint in Pytorch - Towards AI
https://pub.towardsai.net/understand-gradient-checkpoint-in-pytorch-df85511007e1
All Sources
discuss.pytorch
docs.pytorch
github
medium
stackoverflow
pub.towardsai

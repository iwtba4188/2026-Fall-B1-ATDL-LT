import argparse
import copy
import datetime
import os
import random

import numpy as np
import yaml
from act_matching_resnet20 import permute_model_resnet
from act_matching_vgg11 import permute_model
from models import *
from progressive import (
    assert_density,
    grow_mask,
    initialize_random_mask,
    mask_retention_statistics,
    mask_statistics,
    snapshot_masks,
    transfer_mask,
    validate_mask_compatibility,
)
from pruning import prune_model
from resnet50_src.act_matching_resnet50 import permute_model_resnet50
from run_logging import finish_run, log_metrics, start_run
from torch.utils.flop_counter import FlopCounterMode
from training import *
from utils import *


def train_model_a_b(seed, config):
    """
    function to train model_a_dense, model_a_sparse (imp) and model_b dense,
    and saves all checkpoints.
    """

    results = {}

    current_time = datetime.datetime.now().strftime("%d %b %Y %H:%M:%S")

    # Check selected backend is supported
    assert hasattr(torch.backends, config["device"]), (
        f"Device {config['device']} not supported"
    )
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    start_run(
        f"{config['model_type']}_seed_{seed}",
        {
            "project": config["project"],
            "sparsity": config["pruning"]["sparsity"],
            "width": config["width_multiplier"],
            "seed": seed,
        },
        config["save_path"],
    )

    ############################################################## Dense training: Model A ########################################################
    model_A_dense = get_model(config)
    trainer(
        model_A_dense,
        config["optimizer"],
        config["epochs"],
        config["batch_size"],
        config["device"],
        "Model_A_Dense",
        sparse=False,
        config=config,
    )
    train_dl, test_dl = cifar_dataloader(config["batch_size"], config)

    results["A_dense"] = evaluate(model_A_dense, test_dl, config["device"])
    print(f"A_dense: {results['A_dense'][0]:.2f}%, {results['A_dense'][1]:.4f}")

    if config["model_type"] == "ResNet50":
        results["A_dense_top5"] = evaluate_top5(
            model_A_dense, test_dl, config["device"]
        )
        print(
            f"A_dense_top5: {results['A_dense_top5'][0]:.2f}%, Loss: {results['A_dense_top5'][1]:.4f}"
        )
    ###############################################################################################################################################

    ############################################################## Pruning Model A after training #################################################
    if config["model_type"] in ("ResNet", "VGG11", "ResNet50", "LeNet"):
        model_A_sparse = copy.deepcopy(model_A_dense)
        if config["model_type"] == "ResNet" or config["model_type"] == "ResNet50":
            sparsity_tolerance = config["pruning"]["tolerance_resnet20"]
        else:
            sparsity_tolerance = config["pruning"]["tolerance_vgg11"]
        prune_model(
            config,
            model=model_A_sparse,
            target_sparsity=config["pruning"]["sparsity"],
            optimizer_config=config["optimizer"],
            prune_epochs=config["pruning"]["prune_epochs"],
            initial_lr=config["pruning"]["prune_lr"],
            batch_size=config["batch_size"],
            device=device,
            initial_prune_perc=config["pruning"]["prune_perc"],
            train_epochs_per_prune=config["pruning"]["train_epochs_per_prune"],
            sparsity_tolerance=sparsity_tolerance,
        )
        original_model_A_sparse = copy.deepcopy(model_A_sparse)
        torch.save(
            model_A_sparse,
            config["save_path"]
            + "model_a_sparse_sparsity_"
            + str(config["pruning"]["sparsity"])
            + "_seed_"
            + str(config["seed"]),
        )
        print("\n First hooks check \n")
        check_hooks(original_model_A_sparse)

        for name, module in model_A_sparse.named_modules():
            if isinstance(module, nn.Conv2d):
                if (
                    config["model_type"] == "ResNet"
                    or config["model_type"] == "ResNet50"
                    and "shortcut" in name
                    and "downsample" in name
                ):
                    continue
                assert hasattr(module, "weight_mask")
        results["A_sparse"] = evaluate(model_A_sparse, test_dl, config["device"])
        print(f"A_sparse: {results['A_sparse'][0]:.2f}%, {results['A_sparse'][1]:.4f}")
    ###############################################################################################################################################

    ############################################################## Dense training: Model B ########################################################
    model_B_dense = get_model(config)
    trainer(
        model_B_dense,
        config["optimizer"],
        config["model_b_epochs"],
        config["batch_size"],
        config["device"],
        "Model_B_Dense",
        sparse=False,
        config=config,
    )
    results["B_dense"] = evaluate(model_B_dense, test_dl, config["device"])
    print(f"B_dense: {results['B_dense'][0]:.2f}%, {results['B_dense'][1]:.4f}")

    if config["model_type"] == "ResNet50":
        results["B_dense_top5"] = evaluate_top5(
            model_A_dense, test_dl, config["device"]
        )
        print(
            f"B_dense_top5: {results['B_dense_top5'][0]:.2f}%, Loss: {results['B_dense_top5'][1]:.4f}"
        )
    ###############################################################################################################################################

    if config["model_type"] == "ResNet":
        permute_model_0, permuted_model_A_sparse = permute_model_resnet(
            model_A_dense, model_B_dense, model_A_sparse, train_dl, config
        )
        results["A_dense_permuted"] = evaluate(
            permute_model_0, test_dl, config["device"]
        )
        print(
            f"A_dense_permuted: {results['A_dense_permuted'][0]:.2f}%, {results['A_dense_permuted'][1]:.4f}"
        )

        print("\n Pre-reset")
        model_a = evaluate_merged_models(permute_model_0, model_B_dense, 0.5, config)
        train_accuracy, train_loss = evaluate(model_a, train_dl, config["device"])
        test_accuracy, test_loss = evaluate(model_a, test_dl, config["device"])
        print(
            f"(α=0.5): <-- Merged Model\tTrain Accuracy: {train_accuracy}\tTrain Loss: {train_loss:.4f}\t"
            f"Test Accuracy: {test_accuracy}\tTest Loss: {test_loss:.4f}"
        )

        print("\n Post-reset")
        reset_bn_stats(model_a, train_dl)
        results["Merged Model Post reset evaluation"] = evaluate(
            model_a, test_dl, config["device"]
        )
        print(
            f"Merged Model Post reset evaluation: {results['Merged Model Post reset evaluation'][0]:.2f}%, {results['Merged Model Post reset evaluation'][1]:.4f}"
        )

        results["A_sparse_permuted"] = evaluate(
            permuted_model_A_sparse, test_dl, config["device"]
        )
        print(
            f"\n A_sparse_permuted: {results['A_sparse_permuted'][0]:.2f}%, {results['A_sparse_permuted'][1]:.4f}"
        )

        if compare_model_weights(original_model_A_sparse, permuted_model_A_sparse):
            print("\n\n The weights match for all layers.")
        else:
            print("\n\n The weights do not match for all layers.")

    elif config["model_type"] in ("VGG11", "LeNet"):
        #### π(modelA_sparse) ###
        permuted_model_A_sparse, permuted_model_a_dense = permute_model(
            model_A_dense, model_B_dense, model_A_sparse, train_dl, config
        )
        for name, module in permuted_model_A_sparse.named_modules():
            if isinstance(module, nn.Conv2d):
                assert hasattr(module, "weight_mask")
        check_hooks(permuted_model_A_sparse)
        results["A_sparse_permuted"] = evaluate(
            permuted_model_A_sparse, test_dl, config["device"]
        )
        print(
            f"A_sparse_permuted: {results['A_sparse_permuted'][0]:.2f}%, {results['A_sparse_permuted'][1]:.4f}"
        )

        #### Check strength of permutation by interpolating between dense model B and dense permuted model A ####
        results["A_dense_permuted"] = evaluate(
            permuted_model_a_dense, test_dl, config["device"]
        )
        print(
            f"A_dense_permuted: {results['A_dense_permuted'][0]:.2f}%, {results['A_dense_permuted'][1]:.4f}"
        )
        model_a = evaluate_merged_models(
            model_B_dense, permuted_model_a_dense, 0.5, config
        )
        train_accuracy, train_loss = evaluate(model_a, train_dl, config["device"])
        test_accuracy, test_loss = evaluate(model_a, test_dl, config["device"])
        print(
            f"(α=0.5): <-- Merged Model\tTrain Accuracy: {train_accuracy}\tTrain Loss: {train_loss:.4f}\t"
            f"Test Accuracy: {test_accuracy}\tTest Loss: {test_loss:.4f}"
        )

        if compare_model_weights(original_model_A_sparse, permuted_model_A_sparse):
            print("\n\n The weights match for all layers.")
        else:
            print("\n\n The weights do not match for all layers.")

    elif config["model_type"] == "ResNet50":
        num_batches = 50
        batches = []
        for i, (images, labels) in enumerate(train_dl):
            if i >= num_batches:
                break
            batches.append((images, labels))
        batch_size = batches[0][0].shape[0]
        total_data_points = num_batches * batch_size
        print(f"Batch size: {batch_size}")
        print(f"Total data points: {total_data_points}")

        permute_model_0, permuted_model_A_sparse = permute_model_resnet50(
            model_A_dense, model_B_dense, model_A_sparse, batches, config
        )
        torch.save(
            permute_model_0,
            config["save_path"]
            + "permuted_model_0_sparsity_"
            + str(config["pruning"]["sparsity"])
            + "_seed_"
            + str(config["seed"]),
        )
        results["A_dense_permuted"] = evaluate(
            permute_model_0, test_dl, config["device"]
        )
        print(
            f"A_dense_permuted: {results['A_dense_permuted'][0]:.2f}%, {results['A_dense_permuted'][1]:.4f}"
        )

        print("\n Pre-reset")
        model_a = evaluate_merged_models(permute_model_0, model_B_dense, config)
        train_accuracy, train_loss = evaluate(model_a, train_dl, config["device"])
        test_accuracy, test_loss = evaluate(model_a, test_dl, config["device"])
        print(
            f"(α=0.5): <-- Merged Model\tTrain Accuracy: {train_accuracy}\tTrain Loss: {train_loss:.4f}\t"
            f"Test Accuracy: {test_accuracy}\tTest Loss: {test_loss:.4f}"
        )

        print("\n Post-reset")
        reset_bn_stats(model_a, train_dl)
        results["Merged Model Post reset evaluation"] = evaluate(
            model_a, test_dl, config["device"]
        )
        print(
            f"Merged Model Post reset evaluation: {results['Merged Model Post reset evaluation'][0]:.2f}%, {results['Merged Model Post reset evaluation'][1]:.4f}"
        )

        print("\n Another reset (REPAIR)")
        repaired_model_a = repair_merged_model(
            permute_model_0, model_B_dense, model_a, train_dl, config, 0.5
        )
        results["Merged Model with REPAIR"] = evaluate(
            repaired_model_a, test_dl, config["device"]
        )
        print(
            f"Merged Model with REPAIR: {results['Merged Model with REPAIR'][0]:.2f}%, {results['Merged Model with REPAIR'][1]:.4f}"
        )

        ## Ensure this gives the same evaluation as model_A_sparse
        results["A_sparse_permuted"] = evaluate(
            permuted_model_A_sparse, test_dl, config["device"]
        )
        print(
            f"\n A_sparse_permuted: {results['A_sparse_permuted'][0]:.2f}%, {results['A_sparse_permuted'][1]:.4f}"
        )

        if compare_model_weights(original_model_A_sparse, permuted_model_A_sparse):
            print("\n\n The weights match for all layers.")
        else:
            print("\n\n The weights do not match for all layers.")

    torch.save(
        permuted_model_A_sparse,
        config["save_path"]
        + "permuted_model_a_sparse_sparsity_"
        + str(config["pruning"]["sparsity"])
        + "_seed_"
        + str(config["seed"]),
    )
    finish_run()


def train_naive_permuted(seed, k, config):

    results = {}

    current_time = datetime.datetime.now().strftime("%d %b %Y %H:%M:%S")

    # Check selected backend is supported
    assert hasattr(torch.backends, config["device"]), (
        f"Device {config['device']} not supported"
    )
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")

    start_run(
        f"rewind_{k}_{config['model_type']}_seed_{seed}",
        {
            "project": f"{config['project']}_compare",
            "sparsity": config["pruning"]["sparsity"],
            "rewind_point": k,
            "width": config["width_multiplier"],
            "seed": seed,
        },
        config["save_path"],
    )

    # decrease the sparse epochs based on rewind point
    adjusted_sparse_epochs = config["sparse_epochs"] - k
    train_dl, test_dl = cifar_dataloader(config["batch_size"], config)

    ################################################### Sparse training: LTH solution = W_A^{t=k} ⊙ M_A #############################################
    model_A_dense_init_for_LTH = get_model(config)
    load_path_rewound_point_A_for_LTH = f"{config['save_path']}Model_A_Dense_sparsity_{config['pruning']['sparsity']}_seed_{config['seed']}_epoch_{k}"
    model_A_dense_init_for_LTH.load_state_dict(
        torch.load(load_path_rewound_point_A_for_LTH)
    )
    results["model_A_LTH_rewound_point_eval"] = evaluate(
        model_A_dense_init_for_LTH, test_dl, config["device"]
    )
    print(
        f"model_A_LTH_rewound_point_eval: {results['model_A_LTH_rewound_point_eval'][0]:.2f}%, {results['model_A_LTH_rewound_point_eval'][1]:.4f}"
    )

    original_model_A_sparse = torch.load(
        config["save_path"]
        + "model_a_sparse_sparsity_"
        + str(config["pruning"]["sparsity"])
        + "_seed_"
        + str(config["seed"])
    )

    transfer_sparsity_resnet(original_model_A_sparse, model_A_dense_init_for_LTH)
    check_hooks(model_A_dense_init_for_LTH)
    print(
        "Sparsity of the init after transfer sparsity: ",
        calculate_overall_sparsity_from_pth(model_A_dense_init_for_LTH),
    )

    trainer(
        model_A_dense_init_for_LTH,
        config["optimizer"],
        adjusted_sparse_epochs,
        config["batch_size"],
        config["device"],
        "LTH",
        config,
        sparse=True,
        training_type="LTH",
    )
    results["LTH"] = evaluate(model_A_dense_init_for_LTH, test_dl, config["device"])
    print(f"LTH: {results['LTH'][0]:.2f}%, {results['LTH'][1]:.4f}")
    #################################################################################################################################################

    ################################################## Sparse training: Naive solution = W_B^{t=k} ⊙ M_A ############################################
    model_B_dense_init_naive = get_model(config)
    path_rewound_point_B_for_naive_and_permuted = f"{config['save_path']}Model_B_Dense_sparsity_{config['pruning']['sparsity']}_seed_{config['seed']}_epoch_{k}"
    model_B_dense_init_naive.load_state_dict(
        torch.load(path_rewound_point_B_for_naive_and_permuted)
    )

    results["model_B_naive_rewound_point_eval"] = evaluate(
        model_B_dense_init_naive, test_dl, config["device"]
    )
    print(
        f"model_B_naive_rewound_point_eval: {results['model_B_naive_rewound_point_eval'][0]:.2f}%, {results['model_B_naive_rewound_point_eval'][1]:.4f}"
    )

    transfer_sparsity_resnet(original_model_A_sparse, model_B_dense_init_naive)
    check_hooks(model_B_dense_init_naive)
    print(
        "Sparsity of the init after transfer sparsity: ",
        calculate_overall_sparsity_from_pth(model_B_dense_init_naive),
    )

    trainer(
        model_B_dense_init_naive,
        config["optimizer"],
        adjusted_sparse_epochs,
        config["batch_size"],
        config["device"],
        "Model_B_Naive",
        config,
        sparse=True,
        training_type="naive",
    )

    results["B_naive"] = evaluate(model_B_dense_init_naive, test_dl, config["device"])
    print(f"B_naive: {results['B_naive'][0]:.2f}%, {results['B_naive'][1]:.4f}")
    ###############################################################################################################################################

    ################################################# Sparse training: Permuted solution = W_B^{t=k} ⊙ π(M_A) ####################################
    path_rewound_point_B_for_naive_and_permuted = f"{config['save_path']}Model_B_Dense_sparsity_{config['pruning']['sparsity']}_seed_{config['seed']}_epoch_{k}"
    with FlopCounterMode(display=False) as mode:
        model_B_dense_init_perm = get_model(config)
        model_B_dense_init_perm.load_state_dict(
            torch.load(path_rewound_point_B_for_naive_and_permuted)
        )
        ## Ensure this gives the same evaluation as the rewound point from above. ###
        results["model_B_perm_rewound_point_eval"] = evaluate(
            model_B_dense_init_perm, test_dl, config["device"]
        )
        print(
            f"model_B_perm_rewound_point_eval: {results['model_B_perm_rewound_point_eval'][0]:.2f}%, {results['model_B_perm_rewound_point_eval'][1]:.4f}"
        )

        permuted_model_A_sparse = torch.load(
            config["save_path"]
            + "permuted_model_a_sparse_sparsity_"
            + str(config["pruning"]["sparsity"])
            + "_seed_"
            + str(config["seed"])
        )
        transfer_sparsity_resnet(permuted_model_A_sparse, model_B_dense_init_perm)
        check_hooks(model_B_dense_init_perm)
        print(
            "Sparsity of the init after transfer sparsity: ",
            calculate_overall_sparsity_from_pth(model_B_dense_init_perm),
        )

        trainer(
            model_B_dense_init_perm,
            config["optimizer"],
            adjusted_sparse_epochs,
            config["batch_size"],
            config["device"],
            "Model_B_Permuted",
            config,
            sparse=True,
            training_type="permuted",
        )

        results["B_permuted"] = evaluate(
            model_B_dense_init_perm, test_dl, config["device"]
        )
        print(
            f"B_permuted: {results['B_permuted'][0]:.2f}%, {results['B_permuted'][1]:.4f}"
        )

    print("=" * 50)
    print(f"Total FLOPs: {mode.get_total_flops()}")
    print(f"Global FLOPs: {mode.get_flop_counts()['Global']}")
    ###############################################################################################################################################
    finish_run()


def train_progressive(seed, config):
    """Run progressive mask growth with a transferred or random initial mask."""
    progressive_config = config.get("progressive", {})
    targets = progressive_config.get("target_densities", [])
    if not targets:
        raise ValueError(
            "progressive.target_densities must contain at least one target"
        )
    if any(not 0.0 <= target <= 1.0 for target in targets):
        raise ValueError("progressive.target_densities must be in [0, 1]")
    if targets != sorted(targets):
        raise ValueError(
            "progressive.target_densities must be monotonically increasing"
        )

    source_path = (
        config["save_path"]
        + "model_a_sparse_sparsity_"
        + str(config["pruning"]["sparsity"])
        + "_seed_"
        + str(seed)
    )
    initial_mask = progressive_config.get("initial_mask", "model_a")
    if initial_mask not in {"model_a", "random"}:
        raise ValueError("progressive.initial_mask must be 'model_a' or 'random'")

    source_model = torch.load(source_path, map_location=config["device"])
    model_b = get_model(config)
    validate_mask_compatibility(source_model, model_b, config["model_type"])
    source_active, _ = mask_statistics(source_model, config["model_type"])
    initial_mask_generator = torch.Generator()
    initial_mask_generator.manual_seed(
        int(progressive_config.get("initial_mask_seed", seed))
    )
    if initial_mask == "model_a":
        transfer_mask(source_model, model_b, config["model_type"])
    else:
        initialize_random_mask(
            model_b,
            config["model_type"],
            source_active,
            generator=initial_mask_generator,
        )
    initial_masks = snapshot_masks(model_b, config["model_type"])

    start_run(
        f"progressive_{initial_mask}_{config['model_type']}_seed_{seed}",
        {
            "project": f"{config['project']}_progressive",
            "method": "progressive",
            "initial_mask": initial_mask,
            "target_densities": targets,
            "seed": seed,
        },
        config["save_path"],
    )
    train_dl, test_dl = cifar_dataloader(config["batch_size"], config)
    generator = torch.Generator()
    generator.manual_seed(int(progressive_config.get("growth_seed", seed)))
    selection = progressive_config.get("selection", "magnitude")

    for stage, target_density in enumerate(targets):
        active, total = mask_statistics(model_b, config["model_type"])
        current_density = active / total
        if target_density < current_density:
            raise ValueError(
                f"Stage {stage} target density {target_density} is below "
                f"current density {current_density}"
            )
        if selection == "gradient":
            _compute_progressive_gradients(model_b, train_dl)
        grow_mask(
            model_b,
            config["model_type"],
            target_density,
            selection=selection,
            generator=generator,
        )
        assert_density(model_b, config["model_type"], target_density)
        checkpoint = (
            f"{config['save_path']}progressive_stage_{stage}_density_"
            f"{target_density}_{initial_mask}_seed_{seed}.pth"
        )
        torch.save(model_b.state_dict(), checkpoint)
        log_metrics(
            {
                "Progressive Stage": stage,
                "Progressive Density": target_density,
                "Progressive Active Weights": int(round(target_density * total)),
            }
        )
        trainer(
            model_b,
            config["optimizer"],
            progressive_config.get("epochs_per_stage", 1),
            config["batch_size"],
            config["device"],
            "Progressive",
            config,
            sparse=True,
            training_type="progressive",
        )
        evaluate_result = evaluate(model_b, test_dl, config["device"])
        print(
            f"Progressive stage {stage}: density={target_density:.6f}, "
            f"accuracy={evaluate_result[0]:.2f}%"
        )

    grown_masks = snapshot_masks(model_b, config["model_type"])
    final_target = config["pruning"]["sparsity"]
    prune_model(
        config,
        model=model_b,
        target_sparsity=final_target,
        optimizer_config=config["optimizer"],
        prune_epochs=config["pruning"]["prune_epochs"],
        initial_lr=config["pruning"]["prune_lr"],
        batch_size=config["batch_size"],
        device=torch.device(config["device"] if torch.cuda.is_available() else "cpu"),
        initial_prune_perc=config["pruning"]["prune_perc"],
        train_epochs_per_prune=config["pruning"]["train_epochs_per_prune"],
        sparsity_tolerance=config["pruning"].get(
            "tolerance_resnet20"
            if config["model_type"] in ("ResNet", "ResNet50")
            else "tolerance_vgg11",
            0.005,
        ),
    )
    retention = mask_retention_statistics(
        model_b,
        config["model_type"],
        initial_masks,
        grown_masks,
    )
    log_metrics(
        {
            "Progressive Initial Active": retention["initial_active"],
            "Progressive Added Active": retention["added_active"],
            "Progressive Final Active": retention["final_active"],
            "Progressive Initial Retained": retention["initial_retained"],
            "Progressive Added Retained": retention["added_retained"],
            "Progressive Initial Retention": retention["initial_retention"],
            "Progressive Added Retention": retention["added_retention"],
            "Progressive Initial Retention Percent": (
                retention["initial_retention"] * 100
            ),
            "Progressive Added Retention Percent": retention["added_retention"] * 100,
        }
    )
    print(
        "Progressive mask retention: "
        f"initial={retention['initial_retention']:.2%}, "
        f"added={retention['added_retention']:.2%}"
    )
    result = evaluate(model_b, test_dl, config["device"])
    torch.save(
        model_b,
        config["save_path"] + f"model_b_progressive_{initial_mask}_seed_{seed}.pth",
    )
    print(f"Progressive final accuracy: {result[0]:.2f}%")
    finish_run()


def _compute_progressive_gradients(model, train_dl):
    """Compute one loss gradient for gradient-based mask growth."""
    was_training = model.training
    model.eval()
    model.zero_grad(set_to_none=True)
    inputs, targets = next(iter(train_dl))
    device = next(model.parameters()).device
    loss = torch.nn.CrossEntropyLoss()(model(inputs.to(device)), targets.to(device))
    loss.backward()
    model.train(was_training)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", type=str)
    parser.add_argument("-seed", type=int)
    parser.add_argument("-pretrain", type=str)
    parser.add_argument("--rewind", type=int)
    parser.add_argument(
        "--method", choices=("baselines", "progressive"), default="baselines"
    )
    parser.add_argument("-slurm_tmpdir", type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["seed"] = args.seed

    config["save_path"] = (
        f"{config['save_path']}width_{config['width_multiplier']}/sparsity_{config['pruning']['sparsity'] * 100}/"
    )
    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"], exist_ok=True)
    print(config["save_path"])

    print(config)
    torch.manual_seed(args.seed)  # set the random seed
    np.random.seed(args.seed)  # set the random seed

    print(args.pretrain)
    if args.pretrain == "True":
        train_model_a_b(args.seed, config)
    elif args.pretrain == "False" and args.method == "baselines":
        train_naive_permuted(args.seed, args.rewind, config)
    elif args.pretrain == "False" and args.method == "progressive":
        with FlopCounterMode(display=False) as mode:
            train_progressive(args.seed, config)
        print("=" * 50)
        print(f"Total FLOPs: {mode.get_total_flops()}")
        print(f"Global FLOPs: {mode.get_flop_counts()['Global']}")

        log_metrics(
            {
                "Total FLOPs": mode.get_total_flops(),
                "Global FLOPs": mode.get_flop_counts()["Global"],
            }
        )
    else:
        raise ValueError("incorrect argument")

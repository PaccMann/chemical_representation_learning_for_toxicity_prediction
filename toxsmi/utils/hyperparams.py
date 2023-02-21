from paccmann_predictor.utils.hyperparams import LOSS_FN_FACTORY

from .wrappers import BCEIgnoreNaN, GenericIgnoreNaN

LOSS_FN_FACTORY.update(
    {
        "binary_cross_entropy_ignore_nan_and_sum": BCEIgnoreNaN("sum"),
        "binary_cross_entropy_ignore_nan_and_mean": BCEIgnoreNaN("mean"),
        "mse_ignore_nan_and_mean": GenericIgnoreNaN(loss="mse", reduction="mean"),
        "mse_ignore_nan_and_sum": GenericIgnoreNaN(loss="mse", reduction="sum"),
        "l1_ignore_nan_and_mean": GenericIgnoreNaN(loss="l1", reduction="mean"),
        "l1_ignore_nan_and_sum": GenericIgnoreNaN(loss="l1", reduction="sum"),
    }
)

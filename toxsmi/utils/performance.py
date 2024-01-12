import json
import logging
import os
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    fbeta_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    roc_curve,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("performance_logger")
logger.setLevel(logging.INFO)


def binarize_predictions(
    predictions: np.array, labels: np.array, return_youden: bool = False
):
    """
    Binarizes predictions based on Youden's index.

    Args:
        predictions: A 1D np.array with continuous predictions.
        labels: A 1D np.array with discrete labels.
        return_youden: Whether the threshold is returned.

    Returns:
        A 1D np.array with binarized predictions.
    """
    fpr, tpr, thresholds = roc_curve(labels, predictions)
    youden_thresh = thresholds[np.argmax(tpr - fpr)]
    bin_preds = predictions > youden_thresh
    if return_youden:
        return bin_preds, youden_thresh
    else:
        return bin_preds


class PerformanceLogger:
    def __init__(
        self,
        task: str,
        model_path: str,
        epochs: int,
        train_batches: int,
        test_batches: int,
        task_names: List[str],
        beta: float = 1,
    ):
        if task == "binary_classification":
            self.report = self.performance_report_binary_classification
            self.inference_report = self.inference_report_binary_classification
            self.metric_initializer("roc_auc", 0)
            self.metric_initializer("accuracy", 0)
            self.metric_initializer("precision_recall", 0)
            self.metric_initializer("f1", 0)
            self.task_final_report = self.final_report_binary_classification
        elif task == "regression":
            self.report = self.performance_report_regression
            self.inference_report = self.inference_report_regression
            self.metric_initializer("rmse", 10**9)
            self.metric_initializer("mae", 10**9)
            self.metric_initializer("pearson", -1)
            self.metric_initializer("spearman", -1)
            self.task_final_report = self.final_report_regression
        else:
            raise ValueError(f"Unknown task {task}")
        self.metric_initializer("loss", 10**9)

        self.task = task
        self.task_names = task_names
        self.model_path = model_path
        self.weights_path = os.path.join(model_path, "weights/{}_{}.pt")
        self.epoch = 0
        self.epochs = epochs
        self.train_batches = train_batches
        self.test_batches = test_batches
        self.metrics = []
        # for Fbeta score, only used in classification mode
        self.beta = beta

    def metric_initializer(self, metric: str, value: float):
        setattr(self, metric, value)

    def process_data(self, labels: np.array, preds: np.array):
        # Register full arrays
        self.labels = labels
        self.preds = preds

        # From here, everything will be 1D
        labels = labels.flatten()
        preds = preds.flatten()

        # Remove NaNs from labels and predictions to compute scores
        preds = preds[~np.isnan(labels)]
        labels = labels[~np.isnan(labels)]

        return labels.astype(float), preds.astype(float)

    def performance_report_binary_classification(
        self, labels: np.array, preds: np.array, loss: float, model: Callable
    ):
        labels, preds = self.process_data(labels, preds)

        best = ""
        loss_a = loss / self.test_batches
        fpr, tpr, _ = roc_curve(labels, preds)
        roc_auc = auc(fpr, tpr)

        # calculations for visualization plot
        precision, recall, _ = precision_recall_curve(labels, preds)
        # score for precision vs accuracy
        precision_recall = average_precision_score(labels, preds)

        bin_preds, youden = binarize_predictions(preds, labels, return_youden=True)
        report = classification_report(labels, bin_preds, output_dict=True)
        negative_precision = report.get("0.0", {}).get("precision", -1)
        negative_recall = report.get("0.0", {}).get("recall", -1)
        positive_precision = report.get("1.0", {}).get("precision", -1)
        positive_recall = report.get("1.0", {}).get("recall", -1)
        f1 = fbeta_score(
            labels, bin_preds, beta=self.beta, pos_label=1, average="binary"
        )

        logger.info(
            f"\t **** TEST **** Epoch [{self.epoch + 1}/{self.epochs}], "
            f"loss: {loss_a:.5f}, roc_auc: {roc_auc:.5f}, "
            f"avg precision-recall score: {precision_recall:.5f}, "
            f"PosPrecision: {positive_precision:.5f}, "
            f"PosRecall: {positive_recall:.5f}, "
            f"NegPrecision: {negative_precision:.5f}, "
            f"NegRecall: {negative_recall:.5f}, "
            f"F1 ({self.beta}): {f1:.5f}"
        )
        info = {
            "test_loss": loss_a,
            "best_loss": self.loss,
            "test_auc": roc_auc,
            "best_auc": self.roc_auc,
            "test_precision_recall": precision_recall,
            "best_precision_recall": self.precision_recall,
            "test_f1": f1,
            "best_f1": self.f1,
        }
        self.metrics.append(info)
        if roc_auc > self.roc_auc:
            self.roc_auc = roc_auc
            self.save_model(model, "ROC-AUC", "best", value=roc_auc)
            best = "ROC-AUC"
        if precision_recall > self.precision_recall:
            self.precision_recall = precision_recall
            self.save_model(model, "Precision-Recall", "best", value=precision_recall)
            best = "Precision-Recall"
        if f1 > self.f1:
            self.f1 = f1
            self.save_model(model, "F1", "best", value=f1)
            best = "F1"
        if loss_a < self.loss:
            self.loss = loss_a
            self.save_model(model, "loss", "best", value=loss_a)
            best = "Loss"
        return best

    def performance_report_regression(
        self, labels: np.array, preds: np.array, loss: float, model: Callable
    ):
        labels, preds = self.process_data(labels, preds)

        best = ""
        loss_a = loss / self.test_batches

        pearson = float(pearsonr(labels, preds)[0])
        spearman = float(spearmanr(labels, preds)[0])
        rmse = float(np.sqrt(mean_squared_error(labels, preds)))
        mae = float(mean_absolute_error(labels, preds))

        logger.info(
            f"\t **** TEST **** Epoch [{self.epoch + 1}/{self.epochs}], "
            f"loss: {loss_a:.5f}, RMSE: {rmse:.5f}, MAE: {mae:.5f},"
            f"Pearson: {pearson:.5f}, Spearman: {spearman:.5f}."
        )
        info = {
            "test_loss": loss_a,
            "best_loss": self.loss,
            "test_rmse": rmse,
            "best_rmse": self.rmse,
            "test_mae": mae,
            "best_mae": self.mae,
            "test_pearson": pearson,
            "best_pearson": self.pearson,
            "test_spearman": spearman,
            "best_spearman": self.spearman,
        }
        self.metrics.append(info)
        if rmse < self.rmse:
            self.rmse = rmse
            self.save_model(model, "RMSE", "best", value=rmse)
            best = "RMSE"
        if mae < self.mae:
            self.mae = mae
            # self.save_model(model, "MAE", "best", value=mae)
            best = "MAE"
        if pearson > self.pearson:
            self.pearson = pearson
            self.save_model(model, "Pearson", "best", value=pearson)
            best = "Pearson"
        if spearman > self.spearman:
            self.spearman = spearman
            # self.save_model(model, "Spearman", "best", value=spearman)
            best = "Spearman"
        if loss_a < self.loss:
            self.loss = loss_a
            self.save_model(model, "loss", "best", value=loss_a)
            best = "Loss"
        return best

    def inference_report_binary_classification(
        self,
        labels: np.array,
        preds: np.array,
    ):
        labels, preds = self.process_data(labels, preds)
        fpr, tpr, _ = roc_curve(labels, preds)
        roc_auc = auc(fpr, tpr)
        bin_preds, youden = binarize_predictions(preds, labels, return_youden=True)
        precision, recall, _ = precision_recall_curve(labels, preds)
        precision_recall = average_precision_score(labels, preds)
        report = classification_report(labels, bin_preds, output_dict=True)
        negative_precision = report.get("0.0", {}).get("precision", -1)
        negative_recall = report.get("0.0", {}).get("recall", -1)
        positive_precision = report.get("1.0", {}).get("precision", -1)
        positive_recall = report.get("1.0", {}).get("recall", -1)
        accuracy = accuracy_score(labels, bin_preds)
        bal_accuracy = balanced_accuracy_score(labels, bin_preds)
        f1 = fbeta_score(labels, bin_preds, beta=0.5, pos_label=1, average="binary")

        info = {
            "roc_auc": roc_auc,
            "f1": f1,
            "youden_threshold": youden,
            "positive_precision": positive_precision,
            "positive_recall": positive_recall,
            "negative_precision": negative_precision,
            "negative_recall": negative_recall,
            "accuracy": accuracy,
            "balanced_accuracy": bal_accuracy,
            "precision_recall_score": precision_recall,
        }
        self.log_dictionary(info)

        return info

    def inference_report_binarized_regression(
        self, labels: np.array, preds: np.array, threshold: float
    ):
        """
        A regression model trained on a regression task, evaluated in a pseduo-
        binarized setting, based on a threshold
        """
        labels, preds = self.process_data(labels, preds)
        bin_labels = (labels > threshold).astype(int)
        bin_preds = (preds > threshold).astype(int)
        report = classification_report(bin_labels, bin_preds, output_dict=True)
        # If the positive or negative key is not inside the report it implies that:
        # 1) All labels were from the same class (positive or negative)
        # 2) All predicitons were from the same class too
        # --> In that case, precision and recall for the other class are set to 0
        negative_precision = report.get("0", {"precision": 0.0})["precision"]
        negative_recall = report.get("0", {"recall": 0.0})["recall"]
        positive_precision = report.get("1", {"precision": 0.0})["precision"]
        positive_recall = report.get("1", {"recall": 0.0})["recall"]
        accuracy = accuracy_score(bin_labels, bin_preds)
        bal_accuracy = balanced_accuracy_score(bin_labels, bin_preds)
        f1 = fbeta_score(bin_labels, bin_preds, beta=0.5, pos_label=1, average="binary")

        info = {
            "f1": f1,
            "fixed_threshold": threshold,
            "positive_precision": positive_precision,
            "positive_recall": positive_recall,
            "negative_precision": negative_precision,
            "negative_recall": negative_recall,
            "accuracy": accuracy,
            "balanced_accuracy": bal_accuracy,
        }
        self.log_dictionary(info)

        return info

    @staticmethod
    def log_dictionary(info: Dict[str, float]):
        logger.info("\t **** TEST ****\n")
        for k, v in info.items():
            logger.info(f"{k}: {v:.5f}")
        logger.info("\n")

    def inference_report_regression(self, labels: np.array, preds: np.array):
        labels, preds = self.process_data(labels, preds)

        pearson = float(pearsonr(labels, preds)[0])
        spearman = float(spearmanr(labels, preds)[0])
        rmse = float(np.sqrt(mean_squared_error(labels, preds)))
        mae = float(mean_absolute_error(labels, preds))
        info = {"rmse": rmse, "mae": mae, "pearson": pearson, "spearman": spearman}
        self.log_dictionary(info)

        return info

    def save_model(self, model: Callable, metric: str, typ: str, value: float = -1.0):
        model.save(self.weights_path.format(typ, metric))
        if typ == "best":
            logger.info(
                f"\t New best performance in {metric}"
                f" with value : {value:.7f} in epoch: {self.epoch}"
            )
            pd.DataFrame(self.preds, columns=self.task_names).to_csv(
                os.path.join(
                    self.model_path, "results", f"{metric}_best_predictions.csv"
                )
            )
            pd.DataFrame(self.labels, columns=self.task_names).to_csv(
                os.path.join(self.model_path, "results", "labels.csv")
            )
            with open(
                os.path.join(self.model_path, "results", f"{metric}_best_metrics.json"),
                "w",
            ) as f:
                json.dump(self.metrics[-1], f)

    def final_report(self):
        self.metric_df = pd.DataFrame(self.metrics)
        logger.info(
            "Overall best performances are: \n \t"
            f"Loss = {self.loss:.4f} in epoch {self.metric_df['test_loss'].idxmin()} "
        )
        self.task_final_report()

    def final_report_binary_classification(self):
        logger.info(
            f"ROC-AUC = {self.roc_auc:.4f} in epoch {self.metric_df['test_auc'].idxmax()} "
        )
        logger.info(
            f"Precision-Recall = {self.precision_recall:.4f} in epoch {self.metric_df['test_precision_recall'].idxmax()} "
        )
        logger.info(
            f"F1 ({self.beta})= {self.f1:.4f} in epoch {self.metric_df['test_f1'].idxmax()} "
        )

    def final_report_regression(self):
        logger.info(
            f"RMSE = {self.rmse:.4f} in epoch {self.metric_df['test_rmse'].idxmin()} "
        )
        logger.info(
            f"MAE = {self.mae:.4f} in epoch {self.metric_df['test_mae'].idxmin()} "
        )
        logger.info(
            f"Pearson = {self.pearson:.4f} in epoch {self.metric_df['test_pearson'].idxmax()} "
        )
        logger.info(
            f"Spearman = {self.spearman:.4f} in epoch {self.metric_df['test_spearman'].idxmax()} "
        )

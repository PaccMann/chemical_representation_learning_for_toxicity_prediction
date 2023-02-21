import json
import logging
import os
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    auc,
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    roc_curve,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("performance_logger")
logger.setLevel(logging.INFO)


class PerformanceLogger:
    def __init__(
        self,
        task: str,
        model_path: str,
        epochs: int,
        train_batches: int,
        test_batches: int,
    ):

        if task == "binary_classification":
            self.report = self.performance_report_binary_classification
            self.metric_initializer("roc_auc", 0)
            self.metric_initializer("accuracy", 0)
            self.metric_initializer("precision_recall", 0)
            self.task_final_report = self.final_report_binary_classification
        elif task == "regression":
            self.report = self.performance_report_regression
            self.metric_initializer("rmse", 10**9)
            self.metric_initializer("mae", 10**9)
            self.metric_initializer("pearson", -1)
            self.metric_initializer("spearman", -1)
            self.task_final_report = self.final_report_regression
        else:
            raise ValueError(f"Unknown task {task}")
        self.metric_initializer("loss", 10**9)

        self.task = task
        self.model_path = model_path
        self.weights_path = os.path.join(model_path, "weights/{}_{}.pt")
        self.epoch = 0
        self.epochs = epochs
        self.train_batches = train_batches
        self.test_batches = test_batches
        self.metrics = []

    def metric_initializer(self, metric: str, value: float):
        setattr(self, metric, value)

    def performance_report_binary_classification(
        self, labels: np.array, preds: np.array, loss: float, model: Callable
    ):
        best = ""
        loss_a = loss / self.test_batches
        fpr, tpr, _ = roc_curve(labels, preds)
        roc_auc = auc(fpr, tpr)

        # calculations for visualization plot
        precision, recall, _ = precision_recall_curve(labels, preds)
        # score for precision vs accuracy
        precision_recall = average_precision_score(labels, preds)

        logger.info(
            f"\t **** TEST **** Epoch [{self.epoch + 1}/{self.epochs}], "
            f"loss: {loss_a:.5f}, , roc_auc: {roc_auc:.5f}, "
            f"avg precision-recall score: {precision_recall:.5f}"
        )
        info = {
            "test_loss": loss_a,
            "best_loss": self.loss,
            "test_auc": roc_auc,
            "best_auc": self.roc_auc,
            "test_precision_recall": precision_recall,
            "best_precision_recall": self.precision_recall,
        }
        self.metrics.append(info)
        self.preds = preds
        self.labels = labels
        if roc_auc > self.roc_auc:
            self.roc_auc = roc_auc
            self.save_model(model, "ROC-AUC", "best", value=roc_auc)
            best = "ROC-AUC"
        if precision_recall > self.precision_recall:
            self.precision_recall = precision_recall
            self.save_model(model, "Precision-Recall", "best", value=precision_recall)
            best = "Precision-Recall"
        if loss_a < self.loss:
            self.loss = loss_a
            self.save_model(model, "loss", "best", value=loss_a)
            best = "Loss"
        return best

    def performance_report_regression(
        self, labels: np.array, preds: np.array, loss: float, model: Callable
    ):
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
        self.preds = preds
        self.labels = labels
        if rmse < self.rmse:
            self.rmse = rmse
            self.save_model(model, "RMSE", "best", value=rmse)
            best = "RMSE"
        if mae < self.mae:
            self.mae = mae
            self.save_model(model, "MAE", "best", value=mae)
            best = "MAE"
        if pearson > self.pearson:
            self.pearson = pearson
            self.save_model(model, "Pearson", "best", value=pearson)
            best = "Pearson"
        if spearman > self.spearman:
            self.spearman = spearman
            self.save_model(model, "Spearman", "best", value=spearman)
            best = "Spearman"
        if loss_a < self.loss:
            self.loss = loss_a
            self.save_model(model, "loss", "best", value=loss_a)
            best = "Loss"
        return best

    def save_model(self, model: Callable, metric: str, typ: str, value: float = -1.0):
        model.save(self.weights_path.format(typ, metric))
        if typ == "best":
            logger.info(
                f"\t New best performance in {metric}"
                f" with value : {value:.7f} in epoch: {self.epoch}"
            )
            pd.DataFrame({"predictions": self.preds, "labels": self.labels}).to_csv(
                os.path.join(
                    self.model_path, "results", f"{metric}_best_predictions.csv"
                )
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

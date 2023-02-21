import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl


def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog("rdApp.error")

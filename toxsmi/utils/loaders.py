import logging

from .io import get_file

logger = logging.getLogger(__name__)


def load_tox21(featurizer='ECFP'):

    import deepchem as dc

    # Featurize Tox21 dataset

    tox21_tasks = [
        'SR-HSE',
        'SR-MMP',
        'SR-ATAD5',
        'NR-PPAR-gamma',
        'NR-ER-LBD',
        'NR-Aromatase',  # original order
        'NR-AR',
        'NR-AR-LBD',
        'SR-ARE',
        'NR-AhR',
        'SR-p53',
        'NR-ER'  # original order
        # deepchem order
        #'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD',
        #'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53'
    ]
    name = 'tox21_gcn.csv'
    PATH = 'ibm.box.com/shared/static/8tn7zkdoulp8logk6xh5n2yyfqztkr82.csv'

    data_path = get_file(name, PATH)
    num_train_samples = 11764
    num_test_samples = 647
    num_total_samples = num_train_samples + num_test_samples

    if featurizer == 'ECFP':
        featurizer = dc.feat.CircularFingerprint(size=1024)
    elif featurizer == 'GraphConv':
        featurizer = dc.feat.ConvMolFeaturizer()
    elif featurizer == 'Weave':
        featurizer = dc.feat.WeaveFeaturizer()
    elif featurizer == 'Raw':
        featurizer = dc.feat.RawFeaturizer()
    elif featurizer == 'AdjacencyConv':
        featurizer = dc.feat.AdjacencyFingerprint(
            max_n_atoms=150, max_valence=6
        )

    loader = dc.data.CSVLoader(
        tasks=tox21_tasks, smiles_field='smiles', featurizer=featurizer
    )
    dataset = loader.featurize(data_path, shard_size=15000)

    # Initialize transformers
    transformers = [
        dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)
    ]

    logger.info('About to transform data')
    for transformer in transformers:
        dataset = transformer.transform(dataset)
    splitter = dc.splits.IndexSplitter()
    train, _, test = splitter.train_valid_test_split(
        dataset,
        frac_train=num_train_samples / num_total_samples,
        frac_valid=0.,
        frac_test=num_test_samples / num_total_samples
    )
    all_dataset = (train, test)

    return tox21_tasks, all_dataset, transformers


def load_organdb_github(featurizer='ECFP'):

    import deepchem as dc

    # Featurize orgndb github dataset

    organdb_git_tasks = [
        'CHR:Adrenal Gland', 'CHR:Bone Marrow', 'CHR:Brain', 'CHR:Eye',
        'CHR:Heart', 'CHR:Kidney', 'CHR:Liver', 'CHR:Lung', 'CHR:Lymph Node',
        'CHR:Mammary Gland', 'CHR:Pancreas', 'CHR:Pituitary Gland',
        'CHR:Spleen', 'CHR:Stomach', 'CHR:Testes', 'CHR:Thymus',
        'CHR:Thyroid Gland', 'CHR:Urinary Bladder', 'CHR:Uterus', 'MGR:Brain',
        'MGR:Kidney', 'MGR:Ovary', 'MGR:Testes', 'SUB:Adrenal Gland',
        'SUB:Bone Marrow', 'SUB:Brain', 'SUB:Heart', 'SUB:Kidney', 'SUB:Liver',
        'SUB:Lung', 'SUB:Spleen', 'SUB:Stomach', 'SUB:Testes', 'SUB:Thymus',
        'SUB:Thyroid Gland'
    ]
    name = 'organdb_github_gcn.csv'
    PATH = 'https://ibm.box.com/shared/static/8du6hdr1wpdq8wdfddjgxx6d3i1mwen6.csv'

    data_path = get_file(name, PATH)
    num_train_samples = 719
    num_test_samples = 128
    num_total_samples = num_train_samples + num_test_samples

    if featurizer == 'ECFP':
        featurizer = dc.feat.CircularFingerprint(size=1024)
    elif featurizer == 'GraphConv':
        featurizer = dc.feat.ConvMolFeaturizer()
    elif featurizer == 'Weave':
        featurizer = dc.feat.WeaveFeaturizer()
    elif featurizer == 'Raw':
        featurizer = dc.feat.RawFeaturizer()
    elif featurizer == 'AdjacencyConv':
        featurizer = dc.feat.AdjacencyFingerprint(
            max_n_atoms=150, max_valence=6
        )

    loader = dc.data.CSVLoader(
        tasks=organdb_git_tasks, smiles_field='smiles', featurizer=featurizer
    )
    dataset = loader.featurize(data_path, shard_size=15000)

    # Initialize transformers
    transformers = [
        dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)
    ]

    logger.info('About to transform data')
    for transformer in transformers:
        dataset = transformer.transform(dataset)
    splitter = dc.splits.IndexSplitter()
    train, _, test = splitter.train_valid_test_split(
        dataset,
        frac_train=num_train_samples / num_total_samples,
        frac_valid=0.,
        frac_test=num_test_samples / num_total_samples
    )
    all_dataset = (train, test)

    return organdb_git_tasks, all_dataset, transformers


def load_organdb_suppl(featurizer='ECFP'):

    import deepchem as dc

    # Featurize orgndb github dataset

    organdb_suppl_tasks = [
        'ACU:Adrenal Gland', 'ACU:Body Weight', 'ACU:Clinical Chemistry',
        'ACU:Clinical Signs', 'ACU:Eye', 'ACU:Food Consumption',
        'ACU:Intestine Large', 'ACU:Intestine Small', 'ACU:Lung',
        'ACU:Mortality', 'ACU:Nose', 'ACU:Ovary', 'ACU:Spleen', 'ACU:Stomach',
        'ACU:Testes', 'ACU:Thymus', 'ACU:Urinalysis', 'ACU:Water Consumption',
        'CHR:Abdominal Cavity', 'CHR:Adrenal Gland', 'CHR:Artery (General)',
        'CHR:Auditory Startle Reflex Habituation', 'CHR:Bile duct',
        'CHR:Blood', 'CHR:Blood vessel', 'CHR:Body Weight', 'CHR:Bone',
        'CHR:Bone Marrow', 'CHR:Brain', 'CHR:Bronchus', 'CHR:Cervix',
        'CHR:Clinical Chemistry', 'CHR:Clinical Signs', 'CHR:Clitoral Gland',
        'CHR:Coagulating Gland', 'CHR:Ear', 'CHR:Epididymis', 'CHR:Esophagus',
        'CHR:Estrous Cycle', 'CHR:Eye', 'CHR:Food Consumption',
        'CHR:Gallbladder', 'CHR:General', 'CHR:Harderian Gland', 'CHR:Heart',
        'CHR:Hematology', 'CHR:Intestine Large', 'CHR:Intestine Small',
        'CHR:Kidney', 'CHR:Lacrimal Gland', 'CHR:Larynx', 'CHR:Liver',
        'CHR:Locomotion', 'CHR:Lung', 'CHR:Lymph Node', 'CHR:Mammary Gland',
        'CHR:Mesentery', 'CHR:Mortality', 'CHR:Motor activity', 'CHR:Nerve',
        'CHR:Nose', 'CHR:Oral Mucosa', 'CHR:Other', 'CHR:Ovary',
        'CHR:Pancreas', 'CHR:Parathyroid', 'CHR:Parathyroid Gland',
        'CHR:Penis', 'CHR:Peritoneum', 'CHR:Pharynx', 'CHR:Pituitary Gland',
        'CHR:Pleura', 'CHR:Preputial Gland', 'CHR:Prostate', 'CHR:Reflexes',
        'CHR:Salivary glands', 'CHR:Seminal Vesicle', 'CHR:Skeletal Muscle',
        'CHR:Skin', 'CHR:Sperm Measure', 'CHR:Sperm morphology',
        'CHR:Spinal cord', 'CHR:Spleen', 'CHR:Stomach', 'CHR:Testes',
        'CHR:Thymus', 'CHR:Thyroid Gland', 'CHR:Tissue NOS', 'CHR:Tongue',
        'CHR:Tooth', 'CHR:Trachea', 'CHR:Uncertain Primary Site', 'CHR:Ureter',
        'CHR:Urethra', 'CHR:Urinalysis', 'CHR:Urinary Bladder', 'CHR:Uterus',
        'CHR:Vagina', 'CHR:Water Consumption', "CHR:Zymbal's Gland",
        'CHR:[Not In List]', 'DEV:Abdominal Cavity', 'DEV:Adrenal Gland',
        'DEV:Age Landmark', 'DEV:Aorta', 'DEV:Aortic arch', 'DEV:Bladder',
        'DEV:Blood', 'DEV:Blood vessel', 'DEV:Body Weight', 'DEV:Bone',
        'DEV:Brain', 'DEV:Clinical Chemistry', 'DEV:Clinical Signs',
        'DEV:Clitoral Gland', 'DEV:Coordination', 'DEV:Developmental Landmark',
        'DEV:Diaphragm', 'DEV:Ductus arteriosus', 'DEV:Ear', 'DEV:Epididymis',
        'DEV:Esophagus', 'DEV:Estrous Cycle', 'DEV:Eye', 'DEV:Face',
        'DEV:Food Consumption', 'DEV:Gallbladder', 'DEV:General', 'DEV:Gonad',
        'DEV:Great vessels', 'DEV:Heart', 'DEV:Hematology',
        'DEV:Innominate artery', 'DEV:Interparietal', 'DEV:Intestine Large',
        'DEV:Intestine Small', 'DEV:Intestines', 'DEV:Kidney', 'DEV:Limb',
        'DEV:Liver', 'DEV:Locomotion', 'DEV:Lung', 'DEV:Lymph Node',
        'DEV:Mammary Gland', 'DEV:Maternal Wastage', 'DEV:Mesentery',
        'DEV:Mortality', 'DEV:Motor activity', 'DEV:Mouth / Jaw', 'DEV:Nasal',
        'DEV:Nose', 'DEV:Offspring Survival-Early',
        'DEV:Offspring Survival-Late', 'DEV:Other', 'DEV:Ovary', 'DEV:Oviduct',
        'DEV:Pancreas', 'DEV:Paw / Digit', 'DEV:Penis', 'DEV:Peritoneum',
        'DEV:Placenta', 'DEV:Presphenoid', 'DEV:Prostate',
        'DEV:Pulmonary artery', 'DEV:Radius', 'DEV:Reflexes',
        'DEV:Reproductive Outcome', 'DEV:Reproductive Performance',
        'DEV:Seminal Vesicle', 'DEV:Sexual Developmental Landmark', 'DEV:Skin',
        'DEV:Sperm Measure', 'DEV:Sperm morphology', 'DEV:Spinal cord',
        'DEV:Spleen', 'DEV:Squamosal', 'DEV:Stomach', 'DEV:Subclavian artery',
        'DEV:Testes', 'DEV:Thoracic Cavity', 'DEV:Thymus', 'DEV:Thyroid Gland',
        'DEV:Tissue NOS', 'DEV:Trachea', 'DEV:Trunk', 'DEV:Ulna',
        'DEV:Uncertain Primary Site', 'DEV:Ureter', 'DEV:Urinalysis',
        'DEV:Urinary Bladder', 'DEV:Uterus', 'DEV:Vagina',
        'DEV:Water Consumption', 'DEV:Zygomatic', 'DEV:[Clinical]',
        'DEV:[Not In List]', 'DNT:Active Avoidance', 'DNT:Adrenal Gland',
        'DNT:Age Landmark', 'DNT:Aortic arch', 'DNT:Body Weight', 'DNT:Bone',
        'DNT:Brain', 'DNT:Classical conditioning', 'DNT:Clinical Chemistry',
        'DNT:Clinical Signs', 'DNT:Coordination', 'DNT:Delayed Alternation',
        'DNT:Developmental Landmark', 'DNT:Ductus arteriosus',
        'DNT:Epididymis', 'DNT:Esophagus', 'DNT:Estrous Cycle', 'DNT:Eye',
        'DNT:Food Consumption', 'DNT:General', 'DNT:Heart', 'DNT:Hematology',
        'DNT:Instrumental conditioning', 'DNT:Intestine Large', 'DNT:Kidney',
        'DNT:Liver', 'DNT:Locomotion', 'DNT:Lymph Node',
        'DNT:Maternal Wastage', 'DNT:Maze', 'DNT:Mortality',
        'DNT:Motor activity', 'DNT:Mouth / Jaw', 'DNT:Nerve', 'DNT:Nose',
        'DNT:Offspring Survival-Early', 'DNT:Offspring Survival-Late',
        'DNT:Other', 'DNT:Ovary', 'DNT:Passive Avoidance',
        'DNT:Pituitary Gland', 'DNT:Reflexes', 'DNT:Reproductive Outcome',
        'DNT:Reproductive Performance', 'DNT:Sexual Developmental Landmark',
        'DNT:Spinal cord', 'DNT:Spleen', 'DNT:Stomach', 'DNT:Testes',
        'DNT:Thymus', 'DNT:Thyroid Gland', 'DNT:Tissue NOS', 'DNT:Tooth',
        'DNT:Trachea', 'DNT:Water Consumption', 'DNT:[Clinical]',
        'DNT:[Not In List]', 'MGR:Adrenal Gland', 'MGR:Age Landmark',
        'MGR:Auditory Startle Reflex Habituation', 'MGR:Blood',
        'MGR:Body Weight', 'MGR:Bone', 'MGR:Bone Marrow', 'MGR:Brain',
        'MGR:Cervix', 'MGR:Clinical Chemistry', 'MGR:Clinical Signs',
        'MGR:Coagulating Gland', 'MGR:Coordination',
        'MGR:Developmental Landmark', 'MGR:Diaphragm', 'MGR:Ear',
        'MGR:Epididymis', 'MGR:Esophagus', 'MGR:Estrous Cycle',
        'MGR:Estrous cycle length', 'MGR:Eye', 'MGR:Food Consumption',
        'MGR:General', 'MGR:Gonad', 'MGR:Hair Growth', 'MGR:Heart',
        'MGR:Hematology', 'MGR:Intestine Large', 'MGR:Intestine Small',
        'MGR:Kidney', 'MGR:Limb', 'MGR:Liver', 'MGR:Locomotion', 'MGR:Lung',
        'MGR:Lymph Node', 'MGR:Mammary Gland', 'MGR:Maternal Wastage',
        'MGR:Maze', 'MGR:Mortality', 'MGR:Motor activity', 'MGR:Mouth / Jaw',
        'MGR:Nerve', 'MGR:Nose', 'MGR:Offspring Survival-Early',
        'MGR:Offspring Survival-Late', 'MGR:Other', 'MGR:Ovary',
        'MGR:Pancreas', 'MGR:Parathyroid', 'MGR:Parathyroid Gland',
        'MGR:Paw / Digit', 'MGR:Penis', 'MGR:Pituitary Gland', 'MGR:Placenta',
        'MGR:Prostate', 'MGR:Reflexes', 'MGR:Reproductive Outcome',
        'MGR:Reproductive Performance', 'MGR:Salivary glands',
        'MGR:Seminal Vesicle', 'MGR:Sexual Developmental Landmark',
        'MGR:Skeletal Muscle', 'MGR:Skin', 'MGR:Sperm Measure',
        'MGR:Sperm morphology', 'MGR:Spinal cord', 'MGR:Spleen', 'MGR:Stomach',
        'MGR:Testes', 'MGR:Thymus', 'MGR:Thyroid Gland', 'MGR:Tissue NOS',
        'MGR:Tooth', 'MGR:Trunk', 'MGR:Ureter', 'MGR:Urinalysis',
        'MGR:Urinary Bladder', 'MGR:Uterus', 'MGR:Vagina',
        'MGR:Water Consumption', 'MGR:[Clinical]', 'MGR:[Not In List]',
        'NEU:Active Avoidance', 'NEU:Body Weight', 'NEU:Brain',
        'NEU:Clinical Chemistry', 'NEU:Clinical Signs', 'NEU:Estrous Cycle',
        'NEU:Food Consumption', 'NEU:Hematology',
        'NEU:Instrumental conditioning', 'NEU:Liver', 'NEU:Locomotion',
        'NEU:Maze', 'NEU:Passive Avoidance', 'NEU:Reflexes',
        'NEU:Reproductive Performance', 'NEU:Spleen', 'OTH:Adrenal Gland',
        'OTH:Body Weight', 'OTH:Clinical Chemistry', 'OTH:Clinical Signs',
        'OTH:Epididymis', 'OTH:Food Consumption', 'OTH:General', 'OTH:Heart',
        'OTH:Kidney', 'OTH:Liver', 'OTH:Locomotion', 'OTH:Lung',
        'OTH:Mortality', 'OTH:Offspring Survival-Early', 'OTH:Ovary',
        'OTH:Pancreas', 'OTH:Penis', 'OTH:Pituitary Gland', 'OTH:Prostate',
        'OTH:Reflexes', 'OTH:Reproductive Performance', 'OTH:Seminal Vesicle',
        'OTH:Sexual Developmental Landmark', 'OTH:Skeletal Muscle',
        'OTH:Spleen', 'OTH:Testes', 'OTH:Thyroid Gland', 'OTH:Urinalysis',
        'OTH:[Not In List]', 'REP:Abdominal Cavity', 'REP:Adrenal Gland',
        'REP:Age Landmark', 'REP:Blood vessel', 'REP:Body Weight',
        'REP:Clinical Chemistry', 'REP:Clinical Signs',
        'REP:Coagulating Gland', 'REP:Developmental Landmark',
        'REP:Epididymis', 'REP:Estrous Cycle', 'REP:Estrous cycle length',
        'REP:Food Consumption', 'REP:General', 'REP:Hematology',
        'REP:Intestine Large', 'REP:Intestine Small', 'REP:Kidney',
        'REP:Liver', 'REP:Lung', 'REP:Lymph Node', 'REP:Mammary Gland',
        'REP:Maternal Wastage', 'REP:Mortality',
        'REP:Offspring Survival-Early', 'REP:Offspring Survival-Late',
        'REP:Ovary', 'REP:Penis', 'REP:Pituitary Gland', 'REP:Prostate',
        'REP:Reproductive Outcome', 'REP:Reproductive Performance',
        'REP:Seminal Vesicle', 'REP:Sexual Developmental Landmark', 'REP:Skin',
        'REP:Sperm Measure', 'REP:Sperm morphology', 'REP:Spleen',
        'REP:Stomach', 'REP:Testes', 'REP:Thyroid Gland', 'REP:Tissue NOS',
        'REP:Urinalysis', 'REP:Uterus', 'REP:Vagina', 'REP:Water Consumption',
        'REP:[Clinical]', 'REP:[Not In List]', 'SAC:Abdominal Cavity',
        'SAC:Adrenal Gland', 'SAC:Blood', 'SAC:Blood vessel',
        'SAC:Body Weight', 'SAC:Bone', 'SAC:Bone Marrow', 'SAC:Brain',
        'SAC:Clinical Chemistry', 'SAC:Clinical Signs', 'SAC:Epididymis',
        'SAC:Esophagus', 'SAC:Estrous Cycle', 'SAC:Eye',
        'SAC:Food Consumption', 'SAC:Gallbladder', 'SAC:General', 'SAC:Heart',
        'SAC:Hematology', 'SAC:Intestine Large', 'SAC:Intestine Small',
        'SAC:Kidney', 'SAC:Larynx', 'SAC:Liver', 'SAC:Lung', 'SAC:Lymph Node',
        'SAC:Mammary Gland', 'SAC:Mesentery', 'SAC:Mortality', 'SAC:Nerve',
        'SAC:Nose', 'SAC:Offspring Survival-Early', 'SAC:Oral Mucosa',
        'SAC:Other', 'SAC:Ovary', 'SAC:Pancreas', 'SAC:Penis',
        'SAC:Peritoneum', 'SAC:Pituitary Gland', 'SAC:Prostate',
        'SAC:Reproductive Performance', 'SAC:Salivary glands',
        'SAC:Seminal Vesicle', 'SAC:Skeletal Muscle', 'SAC:Skin',
        'SAC:Sperm Measure', 'SAC:Sperm morphology', 'SAC:Spinal cord',
        'SAC:Spleen', 'SAC:Stomach', 'SAC:Testes', 'SAC:Thoracic Cavity',
        'SAC:Thymus', 'SAC:Thyroid Gland', 'SAC:Tissue NOS', 'SAC:Tongue',
        'SAC:Trachea', 'SAC:Urinalysis', 'SAC:Urinary Bladder', 'SAC:Uterus',
        'SAC:Vagina', 'SAC:Water Consumption', 'SAC:[Not In List]',
        'SUB:Abdominal Cavity', 'SUB:Adrenal Gland', 'SUB:Artery (General)',
        'SUB:Blood', 'SUB:Blood vessel', 'SUB:Body Weight', 'SUB:Bone',
        'SUB:Bone Marrow', 'SUB:Brain', 'SUB:Cervix', 'SUB:Clinical Chemistry',
        'SUB:Clinical Signs', 'SUB:Clitoral Gland', 'SUB:Coordination',
        'SUB:Ear', 'SUB:Epididymis', 'SUB:Esophagus', 'SUB:Estrous Cycle',
        'SUB:Estrous cycle length', 'SUB:Eye', 'SUB:Food Consumption',
        'SUB:Gallbladder', 'SUB:Harderian Gland', 'SUB:Heart',
        'SUB:Hematology', 'SUB:Intestine Large', 'SUB:Intestine Small',
        'SUB:Kidney', 'SUB:Lacrimal Gland', 'SUB:Large Intestine',
        'SUB:Larynx', 'SUB:Liver', 'SUB:Locomotion', 'SUB:Lung',
        'SUB:Lymph Node', 'SUB:Mammary Gland', 'SUB:Mesentery',
        'SUB:Mortality', 'SUB:Motor activity', 'SUB:Nerve', 'SUB:Nose',
        'SUB:Offspring Survival-Early', 'SUB:Offspring Survival-Late',
        'SUB:Oral Mucosa', 'SUB:Other', 'SUB:Ovary', 'SUB:Oviduct',
        'SUB:Pancreas', 'SUB:Parathyroid', 'SUB:Parathyroid Gland',
        'SUB:Penis', 'SUB:Peritoneum', 'SUB:Pharynx', 'SUB:Pituitary Gland',
        'SUB:Pleura', 'SUB:Preputial Gland', 'SUB:Prostate', 'SUB:Reflexes',
        'SUB:Reproductive Performance', 'SUB:Salivary glands',
        'SUB:Seminal Vesicle', 'SUB:Skeletal Muscle', 'SUB:Skin',
        'SUB:Sperm Measure', 'SUB:Sperm morphology', 'SUB:Spinal cord',
        'SUB:Spleen', 'SUB:Stomach', 'SUB:Testes', 'SUB:Thoracic Cavity',
        'SUB:Thymus', 'SUB:Thyroid Gland', 'SUB:Tissue NOS', 'SUB:Tongue',
        'SUB:Tooth', 'SUB:Trachea', 'SUB:Uncertain Primary Site', 'SUB:Ureter',
        'SUB:Urethra', 'SUB:Urinalysis', 'SUB:Urinary Bladder', 'SUB:Uterus',
        'SUB:Vagina', 'SUB:Water Consumption', 'SUB:[Not In List]'
    ]
    name = 'organdb_suppl_gcn.csv'
    PATH = 'https://ibm.box.com/shared/static/fwhhndq8svwk4p9dq60zu9mu3f469puk.csv'

    data_path = get_file(name, PATH)
    num_train_samples = 719
    num_test_samples = 128
    num_total_samples = num_train_samples + num_test_samples

    if featurizer == 'ECFP':
        featurizer = dc.feat.CircularFingerprint(size=1024)
    elif featurizer == 'GraphConv':
        featurizer = dc.feat.ConvMolFeaturizer()
    elif featurizer == 'Weave':
        featurizer = dc.feat.WeaveFeaturizer()
    elif featurizer == 'Raw':
        featurizer = dc.feat.RawFeaturizer()
    elif featurizer == 'AdjacencyConv':
        featurizer = dc.feat.AdjacencyFingerprint(
            max_n_atoms=150, max_valence=6
        )

    loader = dc.data.CSVLoader(
        tasks=organdb_suppl_tasks,
        smiles_field='smiles',
        featurizer=featurizer
    )
    dataset = loader.featurize(data_path, shard_size=15000)

    # Initialize transformers
    transformers = [
        dc.trans.BalancingTransformer(transform_w=True, dataset=dataset)
    ]

    logger.info('About to transform data')
    for transformer in transformers:
        dataset = transformer.transform(dataset)
    splitter = dc.splits.IndexSplitter()
    train, _, test = splitter.train_valid_test_split(
        dataset,
        frac_train=num_train_samples / num_total_samples,
        frac_valid=0.,
        frac_test=num_test_samples / num_total_samples
    )
    all_dataset = (train, test)

    return organdb_suppl_tasks, all_dataset, transformers
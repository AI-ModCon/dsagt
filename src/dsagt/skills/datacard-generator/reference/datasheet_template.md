---
language:
- en # ISO language tag
datasheet_version: 0.1.0 # version of the GENESIS datasheet template
name: Datasheet for {DATASET_NAME} # name of the datasheet
tags:
- project:genesis # include on all GENESIS project datasets
- science:lightsource # what kind of science is this for (e.g., materials, biology, lightsource, fusion, climate, etc.)
- keywords: [] # keywords associated with the dataset
- risk:general # indicates level of risk review {general, reviewed, restricted}
---
# Datasheets for Datasets: Contextualizing Scientific Data 

Documenting a scientific dataset in a datasheet requires careful consideration of scope and resolution. 
Scientific data often exists as part of a larger ecosystem of the scientific record: a single experiment may produce multiple datasets, or a dataset may be split into subsets for different analyses. 
Deciding at what level to create a datasheet is not always straightforward—too granular, and you risk an unmanageable number of datasheets; too broad, and important details may be lost. 
Consider how the dataset will be used and cited, who the intended consumers are (e.g., researchers, search engines, AI systems), and what level of documentation will provide meaningful transparency without unnecessary duplication. 
Keep in mind that datasheets are not necessarily one-to-one with dataset DOIs: a datasheet may describe a single dataset, a collection of related datasets, or a versioned release. 
Before you begin, reflect on these relationships and choose a resolution that balances clarity, usability, and sustainability. 
This is not a prescriptive process; it deserves deliberate consideration.

## Before you begin

## Datasheet Structure
 The datasheet is organized into the following sections:
    - Motivation
    - Composition
    - Collection
    - Preprocessing/Cleaning/Labeling
    - Uses
    - Distribution
    - Maintenance
    - Human Subject Research (if applicable)

## Think about your dataset's modalities
Before filling out the datasheet, consider the different data modalities present in your dataset (e.g., sensor time series, microscopy images, simulation results, logbook text). 
Each modality may have unique characteristics, collection methods, and intended uses that should be documented separately. 
When addressing questions in the datasheet, specify which modality you are referring to, especially if the dataset contains multiple modalities. 
This will help ensure clarity and provide a comprehensive understanding of the dataset's structure and content.

## Think about your dataset's intended uses
Before filling out the datasheet, reflect on the intended uses of your dataset. 
Consider the scientific tasks it is designed to support, the research gaps it aims to address, and the theories it seeks to test. 
Understanding the intended applications will help you provide more accurate and relevant information in the datasheet, particularly in sections related to motivation, uses, and limitations. 
This reflection will also guide you in identifying any potential risks or ethical considerations associated with the dataset's use.

## Consider the scope and resolution of your datasheet
Before filling out the datasheet, carefully consider the scope and resolution at which you will document your dataset. 
Decide whether the datasheet will cover a single dataset, a collection of related datasets, or a versioned release. 
Think about the level of detail that will be most useful for your intended audience, balancing clarity and usability with the need to avoid unnecessary duplication. 
This consideration will help ensure that the datasheet effectively communicates the essential information about your dataset while remaining manageable and relevant.

## Consider the those filling out the datasheet
Before filling out the datasheet, think about who will be completing it. 
Ensure that the individuals responsible for authoring the datasheet have a comprehensive understanding of the dataset, including its creation, composition, and intended uses. 
Encourage collaboration among team members who contributed to the dataset to provide accurate and detailed responses. 
This collaborative approach will help ensure that the datasheet reflects a well-rounded perspective and captures all relevant information about the dataset.

### Guidelines
1. Recommend that there be a citable ``scientific record" that contains the Datasheet and Dataset and other elements of the scientific record, including facilities, authors, publications, software, datasets, schemas, and datasheets.
2. Recommend assigning a separate DOI to the Datasheet for independent versioning and authorship tracking, apart from related manuscripts and datasets.
3. Recommend providing a version of this datasheet with information that is suitable for the widest possible distribution and citation allowed under applicable laws. 
If applicable and to the extent permissible in a document for such wide distribution, the datasheet should cite one or more controlled (at various classification levels, proprietary, or export controlled) versions of itself, and individual responses should indicate that the controlled version has different information. 
    Such different ``versions" of the document should have their own unique persistent identifiers. 
4.  Recommend a broad reading of the text of the questions, and provide partial answers if some parts cannot be provided. 
    For example, if a grant from a private foundation lacks an associated grant identifier, the entity's name and a description of the grant should be provided. 
    If providing the name of the grantor or the purpose of creation is not permissible at the wide distribution level, that portion can be left unanswered with an explicit marking of ``not available."
5. item If a combination of responses is not permitted (e.g., indicating the title and the motivation, or the motivation and the funding agency simultaneously may not be permitted) at the wide distribution level, then the topics describing the dataset and its allowed uses are to be given preference over the provenance.
6. In situations where there might be some uncertainty, make sure the datasheet is reviewed by a derivative classifier.

**[[REMOVE "Datasheets for Datasets: Contextualizing Scientific Data" section prior to sharing the Datasheet]]**

# Datasheet for ${DATASET_NAME}

## Dataset Metadata

### Dataset citation:

Include DOI or URL.  Use information that matches a data catalog entry if applicable. Citations in [bibtex format](https://www.bibtex.com/g/bibtex-format/). Please include either a `doi` or `url` field in the citation.

### Name and contact information for the datasheet author(s):

A person or group that was primarily responsible for authoring the datasheet. Provide the Name, [ORCID](https://orcid.org/), affiliation (ROR ID) and email address of the person or group responsible for the datasheet.

## Motivation

### What was the primary purpose for creating this dataset (e.g., to support a specific scientific task, address a research gap, or test a theory)? 
Describe the intended use and, if applicable, any assumptions made about the underlying system (e.g., proxy variables, stationarity, symmetry, theoretical models).

### Was the dataset created for or in the context of an AI application?
Please provide a description of the AI application, including the type of model(s) that will be trained or evaluated using this dataset.

### Who created the dataset (e.g., which team, research group), what are their institutional affiliations, and on behalf of which research entity (e.g., company, institution, organization) was the dataset created? 

Provide the [ROR ID](https://ror.org/) for research entities if available.

### What resources were used? For example, what funding, facility time, computing resources, and datasets were used to create the dataset? Provide answers in the sections below.

#### Facilities:

list the facilities. Provide the facility name and [ROR ID](https://ror.org/) if available.

#### Funding: 

If there are associated grants, please provide the name of the grantor(s) (e.g., federal agency and program office name) and [ROR ID](https://ror.org/) , and the grant name(s) and number(s) (e.g., PAMS Award number and/or lab contract number). List the solicitation numbers and links, or citations. Citations in [bibtex format](https://www.bibtex.com/g/bibtex-format/). Please include either a `doi` or `url` field in the citation.

#### Other Supporting Entities: 

If available, provide a [ROR ID](https://ror.org/), link, or citation for other supporting entities. Citations in [bibtex format](https://www.bibtex.com/g/bibtex-format/). Please include either a `doi` or `url` field in the citation.

### Was the dataset created as part of a larger initiative?  
If so, provide the name for the initiative, e.g., SciDAC, ModelTeam, Genesis.

### Any other comments regarding the motivation for creating the dataset? (optional)
Provide any additional information about the motivation for creating the dataset.

## Composition

### Identify the data modalities contained in the dataset (e.g., sensor time series, microscopy images, simulation results, logbook text). 
If multiple modalities exist, list each one.

### What do the instances within each modality represent (e.g., detector hits, environmental monitoring readings, simulation runs, documents)? Are there multiple types of instances (e.g., beam pulses and cavity field maps; users and their proposals)? 
Provide a description for each modality.

### How many instances are there in total (of each type, if appropriate)?
Provide the number of instances for each modality.

### Does the dataset include all possible instances for each modality or a subset (not necessarily random)? 
If a subset, what is the larger collection (e.g., all experiments at a user facility during a given period)? Is the subset intended to be representative (e.g., energy ranges, geographic sites)? Describe the criteria and validation methods used to verify representativeness.
If it is not representative, please describe why not (e.g., to capture diversity in experimental conditions; due to data access restrictions; because data was intentionally excluded, and why).

### What data does each instance within the dataset consist of (for each modality, if appropriate)? "Raw" data (e.g., unprocessed text or images) or features?
In either case, please provide a description.

### Is any information missing from individual instances? 
If so, please provide a description explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information but may include, e.g., redacted text or missing sensor data.

### Are relationships between individual instances made explicit (e.g., parent-child relationships in a tree structure, protein–protein interaction networks)? 
If so, please describe how these relationships are made explicit.

### Have the data been split, or are there recommended data splits (e.g., training, development/validation, testing)? 
If so, please provide a description of these splits, explaining the rationale behind them.

### Are there any errors, sources of noise, or redundancies in the dataset? 
If so, please provide a description.

### Are external resources (e.g., file stores, calibration information, software packages, code, websites, other datasets) needed to interpret or reuse these data?
If not, respond "No."
Otherwise, for each external resource, provide a description, and if applicable, its access point (e.g., DOIs, link, [bibtex format](https://www.bibtex.com/g/bibtex-format/) citation). Please include either a `doi` or `url` field in the citation.

#### If there are external resources, for each external resource, are there any restrictions (e.g., licenses, fees)?

### Does the dataset contain data that might be considered Controlled Unclassified Information (CUI) or confidential (e.g., Personally Identifiable Information (PII), Confidential Business Information (CBI), pre-publication data, data protected by legal privilege, or data that includes the content of individuals' non-public communications)?
Refer to [Open Digital Rights Language (ODRL) Vocabulary & Expression](https://www.w3.org/TR/odrl-vocab/) and the [Information Model](https://www.w3.org/TR/odrl-model/) for a standard vocabulary to represent statements about the usage of data.

### Does the dataset contain classified or export-controlled information? 
If so, please provide an explanation.

### Any other comments regarding the composition of the dataset? (optional)
Provide any additional information about the composition of the dataset.

## Collection

### How was the data for each instance obtained or generated (e.g., raw experimental measurements from user facilities, processed/physics-ready experimental data, outputs from computational simulations, or data derived from prior datasets)? 
If the data were simulated or derived, provide details on how accuracy and validity were assessed.

### Did the dataset incorporate data obtained through unconventional means (e.g., online crowdsourcing, volunteer-based citizen science)? 
If yes, explain how these contributions were managed, including documentation, acknowledgment, and quality assurance measures.

### For each instrument or facility used to generate and collect the data, what mechanisms or procedures were used to collect the data (e.g., hardware apparatuses or sensors, manual human curation, software programs, software APIs)? 
Provide a description for each instrument or facility used, mechanisms and procedures used to collect the data. Provide relevant details on how these mechanisms or procedures were validated.

### If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)? 
Provide a description of the sampling strategy and any methods used to assess sample representativeness.

### Over what timeframe was the data collected or generated? Does this timeframe align with when the underlying phenomena or events occurred (e.g., recent simulation of historical accelerator configurations)? If not, describe the timeframe of the original events or conditions represented by the data.
Provide single date, range, or approximate date; suggested format YYYY-MM-DD, and, if appropriate, describe alignment with underlying phenomena or events

### Any other comments regarding the dataset collection process? (optional)
Provide any additional information about the collection process for the dataset.

## Preprocessing/cleaning/labeling

### To create the final dataset, was any preprocessing/ cleaning/ labeling of raw data done?

Preprocessing, cleaning, and labeling can include simple pipelines (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values), or more complex workflows including noise reduction and filtering, calibration, reconstruction and event building, simulation post-processing, meta-data enrichment (e.g., adding uncertainty estimates). 
If so, please provide a description of the workflow.

### Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?
If so, please provide a [bibtex format](https://www.bibtex.com/g/bibtex-format/) citation, link, or other access point to the "raw" data. Please include either a `doi` or `url` field in the citation.

### Is the software that was used to preprocess/ clean/ label the data available? 
If so, please provide a [bibtex format](https://www.bibtex.com/g/bibtex-format/) citation, PID, link, or other access point, along with descriptions of any required packages or libraries to run the scripts. Please include either a `doi` or `url` field in the citation.

### Any other comments regarding the preprocessing, cleaning, and labeling workflow for the dataset? (optional)
Provide any additional information about the preprocessing, cleaning, and labeling workflow for the dataset.

## Uses

### Have AI-Readiness assessment tools been used? 
If so, name the tool, and describe the results?

### To what extent are the data prepared for AI? 
Describe any steps taken to make the data suitable for AI applications, such as formatting, normalization, or feature extraction.

### If the data has been used for AI, provide a description.
Highlight how the data was used for AI, including links to code or DOIs illustrating its use.

### What (other) tasks could the dataset be used for?
Provide examples of tasks that the dataset could be used for, beyond its original purpose.

### Are there limitations or characteristics, arising from the collection, preprocessing, or labeling, that may impact future uses of the dataset (e.g., incomplete coverage of parameter space, data quality variations, restrictions due to CUI or export control)? 
If so, please provide a description. Describe mitigation strategies for the risks?

### Are there tasks for which the dataset should not be used? 
Describe tasks not recommended for this dataset's use.

### Were any reviews for safety, cybersecurity, export control, or other considerations conducted?  
If so, please provide a description of these review processes, including the outcomes.

### Are there tutorials or other supporting information to assist a user of these data?  
If so, please provide a description and [bibtex format](https://www.bibtex.com/g/bibtex-format/) citation, PID, link, or other access point. Please include either a `doi` or `url` field in the citation.

### Any other comments on the use of the dataset? (optional)
Provide any additional information about the use of the dataset.

## Distribution

### Will all or parts of the dataset be shared outside the originating laboratory, facility, or site (e.g., with other DOE sites, collaborating institutions, or public repositories)?
If so, please provide a description of how the dataset will be shared, including any access mechanisms (e.g., data repositories, data catalogs, or other data sharing platforms). Provide relevant links or citations in [bibtex format](https://www.bibtex.com/g/bibtex-format/). Please include either a `doi` or `url` field in the citation.

### Indicate the date or timeframe when the dataset was (or will be) made available.
Provide a single date or range; suggested format YYYY-MM-DD.

### Access and reuse restrictions placed on the data: 
Will the dataset be shared under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)? 
If your dataset is split into multiple parts (e.g., training and test sets), you may need to answer separately for each part.
If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.

### Have any third parties imposed IP-based or other restrictions on the data associated with the instances? 
If so, please describe these restrictions and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions.

### Export control restrictions:
Do any export controls or other regulatory restrictions apply to the dataset or to individual instances? 
If so, please describe these restrictions and provide a link or other access point to, or otherwise reproduce, any supporting documentation.

### If the dataset or any individual instances have a classification or restriction, what is the specific level of classification or restriction (e.g., Classified, Unclassified, CUI)? 
If there is no classification or restriction, respond "N/A."

### Any other comments regarding the distribution of the dataset? (optional)
Provide any additional information about the distribution of the dataset.

## Maintenance

### Who will maintain and provide access to the dataset after its creation? 
For example, DOE-supported repository, facility staff, or facility data steward.
Provide responsible parties’ names, roles, contact information, and persistent identifiers (e.g., [ORCID](https://orcid.org/)). 
Indicate any DOE policies or agreements governing maintenance and retention.

### Is there an erratum? 
If so, please provide a link or other access point.

### Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)? 
If so, please describe how often, by whom, and how updates will be communicated to dataset consumers (e.g., mailing list, GitHub)?

### Will older versions of the dataset continue to be supported/hosted/maintained? 
If so, please describe how. If not, please describe how its obsolescence will be communicated to dataset consumers.

### What is the expected lifespan of the dataset? 
Provide a description of the expected lifespan, including any plans for archiving or decommissioning, including stakeholders, communities, and collaborators who will be consulted. If a retention policy applies, please provide a link or other access point to, or otherwise reproduce, the relevant documentation.

### Will the dataset support controlled extensions or updates from collaborators? 
If yes, describe the mechanism for submission, validation, and versioning, and how these changes will be communicated to users.Otherwise, explain why contributions are restricted.

### Any other comments regarding the maintenance of the dataset? (optional)
Provide any additional information about the maintenance of the dataset.

## Human Subject Research

### Does the dataset include personal information (e.g., identifiable data, survey responses)? 
If not, respond "No" and ignore the remainder of the Datasheet.
Otherwise, respond "Yes" and continue.

### Human Subject Research: Dataset Composition

#### Does the dataset identify any subpopulations (e.g., by age, gender, career stage)?
If not, answer "No." Otherwise, please describe how these subpopulations are identified and provide a description of their respective distributions within the dataset.

#### Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset? 
If not, answer "No." Otherwise, please describe how.

#### Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals race or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)? 
If not, answer "No." Otherwise, please provide a description.

### Human Subject Research: Collection

#### Were any ethical review processes conducted (e.g., by an institutional review board)? 
If so, please provide a description of these review processes, including the outcomes, and a link or other access point to any supporting documentation.

#### Were the individuals in question notified about the data collection? 
If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself.

#### Did the individuals in question consent to the collection and use of their data? 
If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented.

#### If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses? 
If so, please provide a description and a link or other access point to the mechanism (if applicable).

#### Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted? 
If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation.

### Human Subject Research: Maintenance

#### Are there applicable limits on the retention of the data associated with the instances (e.g., were the individuals in question told that their data would be retained for a fixed period of time and then deleted)?
If so, please describe these limits and explain how they will be enforced.

### Human Subject Research: Other

#### Any other comments regarding the inclusion of human subject research in the dataset? (optional)
Provide any additional information about the inclusion of human subject research in the dataset.

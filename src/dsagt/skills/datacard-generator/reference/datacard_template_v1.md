---
#YAML Metadata to support datacard metadata:  (Instructions follow)
#----------------Datacard Information-- This section provides information about this datacard and its contents
datacard_info:
    filename: modcon_datacard_${SNAKE_CASE_DATASET_NAME}.md  # Follow this filename convention and enter the filename here
    id:      # PID assigned to this datacard if used.
        type: #DOI | HANDLE | URL | LOCAL | OTHER
        value:
    datacard_access:  #describe any access restrictions for this datacard, or "None - publicly accessible"
    datacard_creation:
        created_date: "${CREATED_DATE_YYYY_MM_DD}" # required
        update_date: "${UPDATE_DATE_YYYY_MM_DD}" # add updates as needed 
        creation_method: #manual | automated | hybrid
        created_by: [] # Contacts for this datacard.  Include at least one  (datacard_contact_person) or organizational contact (datacard_contact_organization) with email.
          - datacard_contact_person:
            name:
            orcid:
            email:
            affiliation:
          - datacard_contact_organization:
              organization_name:
              ror_id:
              email: 
    datacard_template_version: 1.0 
    language: en

#-------------Data Identification-- Required.  This section contains the information needed to link this datacard to the datasets it describes:
data_identifiers:
    name: "${DATASET_NAME}" #provide a single, human readable name even if this datacard describes a collection of data.  Ensure alignment with the datacard filename.
    dataset_id: # Identify the dataset(s) described by this datacard. At least 1 identifier is required.  Persistent, unique identifiers (PIDs) are preferred.
        type: # DOI | HANDLE | URL | LOCAL | OTHER
        value:

#----- Level 1: Discoverable-----
#[Required Level1]:
# Identification
title: "${DATASET_NAME}" 
id:      # ID of the datacard if known. Otherwise this will be provided by the system.
project:  # required - project name or identifier that the dataset is associated with

#--- Data files/objects in dataset

# for published datasets, include one or more dois. For unpublished datasets, provide a landing page url if available, or other identifier such as a local identifier or handle.
identifiers: []  # [required level1]. At least 1 identifier is required.
  - type: # DOI | HANDLE | URL | LOCAL | OTHER
    value:


#[Optional Level1]:
#--- Dataset Structure / Characteristics---
dataset_info:
  modalities: []  # e.g., ["tabular","image","time-series"]
  data_formats: []
  features: [] # list of variables or features in the dataset.
  splits: [] # e.g., ["train", "test", "validation"]
  dataobject_type: dataset #Type of digital object (dataset, aiagent, eval, framework, model, etc.)
  dataset_type: #See OSTI DOE Data Explorer Types 
  
#**OSTI DOE Data Explorer Dataset Types**:
#| Code | Type | Description |
#|------|------|-------------|
#| `GD` | Genome/Genetic Data | DNA/RNA sequences, genetic markers, genomic annotations |
#| `IM` | Image | Photographs, scans, visualizations, microscopy |
#| `ND` | Numeric Data | Measurements, time series, tabular data, sensor readings |
#| `SM` | Specialized Mix | Multiple data types combined |
#| `FP` | Figure/Plot | Charts, graphs, plots as primary deliverable |
#| `I` | Interactive Resource | Web applications, interactive visualizations, dashboards |
#| `MM` | Multimedia | Audio, video, combined media |
#| `MD` | Model | Computational models, simulations, trained ML models |
#| `AS` | Automated Software | Scripts, analysis pipelines, workflows |
#| `IP` | Instrumentation and Protocols | Experimental protocols, instrument specifications |
#| `IG` | Integrated Genomic Resources | Combined genomic databases and tools |

dataset_readiness:
  level: # 1 | 2 | 3
  evaluated_against:  "Genesis Dataset Readiness Model v1.0"
  evaluated_at:
  evaluated_by:
  confidence: # high | medium | low


#--- Lifecycle / Governance---
release_status: # draft | approved | published | deprecated

review_process:
  review_purpose: # a short description of why is the data is being reviewed (e.g., "release", "partner sharing", "IRB",... )
  review_status: # "submitted" | "pending" | "approved" | "declined" | 
  review_institution:
            name: # name of institution conducting the review
            ror_id: # ROR ID of institution conducting the review
  review_comments: 


#------ Level 2: Interoperable and Reusable-------

#[Required Level2]:
#--- Contacts
contact_point:
  entity:
    type: person # person
    person:
      given_name:
      family_name:
      orcid:
      email:
      affiliation:
        entity:
          type: organization # organization
          organization:
            name:
            ror_id:

additional_contacts: []

#--- Access, Permissions and Policy
access_policy:
  # sensitivity_tier classification:
  # tier0_openScience
  # tier1_controlledResearch
  # tier2_proprietary
  # tier3_sensitive_exportControlled
  # tier4_regulated_personal
  # tier5_classified
  sensitivity_tier:
  access_level: # NOTE: currently only accepting "open", open | restricted | controlled - required  NOTE: currently only accepting "open"
  authorization: # none | accountRequired | userAgreement | dataUseAgreement | sponsorApproval | exportControlReview | irbApproval | other
  policy_url:
  policy_text:

#[IF Applicable]:
cui_markings: # see CUI documentation
distribution_statement:  #examples are:  "Distribution A - Approved for public release"; "Distribution D - DoD only"; "Internal DOE use only"
handling_instructions:  #examples are "No foreign dissemination" or "Export-controlled handling required"

security_marking:
  classification:  # unclassified | cui | classified | confidential | Secret | TS | other
  cui_marking:
  distribution_statement:
  handling_instructions:
  declassification:
    review_date:
    authority:

#--- Rights & License---
license:
  spdx_id: # SPDX ID is a standardized license identifier (e.g., CC-BY-4.0, MIT, Apache-2.0)
  name:
  link:

additional_licenses: []

#[Optional Level2]:
# Domain and Purpose
categorization:
  science_domain: "${SCIENCE_DOMAIN}" # required - high level scientific domain or discipline that the dataset is associated with, e.g., physics, chemistry, materials science, etc.
  modalities: []          # e.g., ["tabular","image","time-series"]
  task_category: []       # ML classification
  task_subcategory: [] # ML subcategory classification

# Resources
originating_research_organization:
  entity:
    type: organization # organization
    organization:
      name:
      ror_id:

facilities: []
# list of facilities (as entities, type: organization) where the dataset was collected, processed, stored, or accessed. Examples include research labs, data centers, cloud platforms, supercomputing facilities, etc.
  
fundings:
# list of funding sources that supported the creation, maintenance, or sharing of the dataset. Examples include grants, contracts, in-kind support, etc.
  - funding:
      award_number:
      program:
      funder:
        entity:
          type: organization # organization
          organization:
            name:
            ror_id:

#--- Dataset Responsibility & Credit
dataset_authors: 
  - entity: # at least 1 is required
      type: person # person
      person:
        given_name:
        family_name:
        orcid:
        email:
        affiliation:
          entity:
            type: organization # organization
            organization:
              name:
              ror_id:
    role: creator

dataset_contributors: []  # Acknowlege other contributors beyond dataset authors and include them as entities of type person or organization.

#--- Stewardship & Maintenance---
stewardship:
  level: # projectManaged | repositoryManaged

maintenance:
  update_frequency: # none | ad_hoc | monthly | quarterly | annually | other

# Provenance
#--- Dataset Provenance---
dataset_provenance:
  was_generated_by:
  source_data:
  processing_steps:
  instrumentation:
  simulation_details:

#--- Related Resources (Optional but recommended)
related_resources:
  related_datasets: []
  # - dataset_name:
  #     identifiers:
  #       - type: DOI | URL | LOCAL | OTHER
  #         value:
  #     relationship: isDerivedFrom | isBasedOn | isPartOf | hasPart | references | other
  publications:
    dois: []
    arxiv: []
    urls: []
  software:
  #  - name:
  #    version:
  #    identifiers:
  #       - type: DOI | URL | LOCAL | OTHER
  #         value:
  #     relationship: isDerivedFrom | isBasedOn | isPartOf | hasPart | references | other
  aimodels: []
  #  - name:  #if available, provide the name of the AI model
  #      version:
  #      date accessed: 
  #      identifiers:
  #        - type: DOI | URL | LOCAL | OTHER
  #          value:
  #      relationship: isDerivedFrom | isBasedOn | isPartOf | hasPart | references | other
#-- Methods
#--- Dates
dates:
  data_collection_start:
  data_collection_end:
  issued:
  modified:


#------ Level 3: Understandable & Trustworthy-------

#[If Applicable Level3]:
#--- Semantic / Schema
semantic_layer:
  # Describe formal schema / ontology alignment if available
  # Leave blank if none exists (acceptable at Level 1 or Level 2)
  schema_url:
  ontology_alignment: []
  semantic_context: []
  controlled_vocabularies: []

#--- Data Quality
data_quality:
  completeness:
  known_issues:
  validation_methods:
  noise_characteristics:
  uncertainty_notes:

#--- Integrity / Fixity
integrity:
  checksum_available:  # true | false
  checksum_type:
  fixity_policy:
  versioning_strategy:

#--- AI / ML Usage
ai_usage:
  ai_ready:  # true | false | conditional
  training_use_allowed:  # true | false | conditional
  inference_use_allowed:  # true | false | conditional
  restrictions:
  bias_risks:
  safety_considerations:
  human_review_required:  # true | false

#---- System Level Info:----

#--- Dataset Scale
dataset_counts:
  value:
  category:  # samples | files | records | timesteps | other
  unit:

dataset_storage:
  compressed_bytes:
  unpacked_bytes:

#--- Repository-managed access endpoints---
repository_access:
  populated_by_repository: true
  distributions: []
  # Example:
  #  - distribution:
  #      access_url: {}
  #      download_url:
  #      format:
  #      byte_size:
  #      checksum:
  #        type: []
  #        items: []
  data_services: []
  # Example:
  #  - data_service:
  #      type: api | s3 | other
  #      additional_properties: false
  #      properties:
  #        endpoint: {}
  #        service_type: {}
  #        auth_hint: {}
---

### Instructions  
<INSTRUCTIONS: Provide relevant information regarding your dataset in this file.  The information can be a combination of text blocks and values in YAML sections.  For the text, replace all Examples, and [!TODO], and REPLACE: ... placeholder tags with the appropriate information for your dataset. Be sure to remove the header TODO and INSTRUCTION tags once you have completed the datacard.  In each section you can complete YAML values as available.  Required fields are marked.>  
  
<INSTRUCTIONS: Considerations for filling out the datacard: Deciding the appropriate resolution for documenting a scientific dataset in a datacard can be complex. Datacards may describe single or multiple data files, datasets, or versions. Too granular, and there will be too many datacards; too broad, and details may be lost. Consider the datacard's use, audience, and documentation needs to maintain transparency without duplication. Reflect on these relationships to balance clarity, usability, and sustainability.>  
  
<INSTRUCTIONS: The sections and questions in the markdown section of this datacard template are meant to be a guide for the types of information that should be included in a datacard. You can choose to answer all, some, or additional questions as appropriate for your dataset. The goal is to provide enough information for users to understand the dataset and its context, but you can use your judgment to determine what information is most relevant and important to include.>  
  
<INSTRUCTIONS: Data readiness in a shared environment can be generally sorted into three high-level categories:  
Level 1: Discoverable  
Level 2: Interoperable & Reusable (Accessible, reusable, and governed within defined sensitivity, license, and access constraints)  
Level 3: Understandable & Trustworthy (Semantically clear, context-rich, provenance-aware, integrity-supported, and reliable for advanced reuse by humans and machines, including AI workflows)  
Readiness Levels describe usability, interoperability, and governance characteristics.  They do NOT represent dataset quality, scientific merit, or value ranking.  
Prompts for data are generally organized into sections that support these efforts.  Some fields are required to be considered for use at each level and for making datasets available to tools targeting each level.  Requirements are labelled for each level.  
>  
  
<INSTRUCTIONS: metadata_key: [KEY_NAME] tags indicate that the information for the markdown section can be found in the corresponding key in the YAML metadata at the top of this file, and is for use in the human and the automated bi-directional generation from YAML-to-markdown, or markdown-to-YAML. You can choose to either manually copy, or you can leave the placeholders and use an automation tool (such as an LLM) to populate the sections. If you choose to automatically populate the markdown sections from the YAML metadata, make sure to replace metadata_key: [KEY_NAME] tags in each relevant markdown section before sharing the datacard, and note the LLM or aiagent used to do this in the datacard_creation section of the YAML frontmatter metadata.>  
  
  
# Datacard for ${DATASET_NAME}  
**Last Updated**: [!TODO]<REPLACE: YYYY-MM-DD>  
  
**Dataset Readiness Level:** <metadata_key: dataset_readiness.level>  
  
### Machine Usability Snapshot  
| Aspect | Status |  
|--------|--------|  
| AI Ready | Yes/No/Conditional|  
| License Clarity | Yes/No|  
| Machine Access | Yes/No|  
| Checksum / Fixity | Yes/No|  
| Semantic Context | Yes/No|  
  
  
# ---- Level 1: Discoverable ----  
---  
## Identification  
  
  
### Files & Structure  
[!TODO] [required level1] <REPLACE: Summarize dataset organization, formats, and relationships between files.>  
  
---  
## Description  
  
### Dataset Description [required]  
[!TODO] <REPLACE: Provide a concise description of the dataset, including its purpose, scope, and context.>  
  
### Keywords [strongly recommended]  
[!TODO] <REPLACE: Provide a comma-separated list of keywords that describe the dataset and can help with discoverability.>  
  
### Citation  
[!TODO] <REPLACE: Provide a recommended citation if known.><metadata_key: dataset_id>
  
---  
  
# ---- Level 2: Interoperable and Reusable ----  
  
### Sharing & Access  
[!TODO] <REPLACE:  Describe the sharing methods and any contact information for access.>  
  
### Security / Marking Considerations  
[!TODO]<Describe classification, CUI marking, distribution limitations, and handling requirements.><metadata_key: security_marking>  
  
---  
### Access and Permissions  
[!TODO] <REPLACE: Describe the dataset`s access posture and any high-level agreements or review constraints. >  
  
### Access conditions  
[!TODO] <REPLACE: Describe any conditions that must be met to access the dataset, such as training requirements, proposal processes, collaboration requirements, data use agreements, etc.><metadata_key: access_policy>  
  
### Release review process  
[!TODO] <REPLACE: Describe the release review process for the dataset, including any institutional reviews, export control reviews, IRB reviews, or other review processes that were conducted before the dataset was released.> <metadata_key: release_status>  
  
---  
## Context  
  
### Domain and Purpose  
[!TODO] <REPLACE: Describe the domain and the key research areas involved in collecting the dataset. Can list below>  
  
### Resources used, including funding and facilities, to create the dataset  
[!TODO] <REPLACE: Provide a list of the resources used to create the dataset, including funding sources, facilities, computing resources, and any other relevant resources. Facilities can include user facilities, national laboratories, research institutions, and other organizations that provided access to equipment, data, or expertise. Funding sources can include government agencies, private foundations, industry partners, and other organizations that provided financial support for the dataset creation. Computing resources can include high-performance computing clusters, cloud computing platforms, and other computational resources used for data processing and analysis. Include [ROR ID](https://ror.org/), grant numbers, contract numbers, or other identifiers as appropriate. Can list below> <metadata_key: facilities> 
  
---  
## Provenance  
  
### Developed by  
[!TODO] <REPLACE: A person or group that was primarily responsible for the creation and design of the dataset. It suggests a leading role, such as a Principal Investigator, in the development of the dataset. If available, provide the Name, [ORCID](https://orcid.org/), affiliation ([ROR ID](https://ror.org/)) and email address of the person or group responsible for the dataset. Can list in dataset_authors below> <metadata_key: dataset_authors> 
  
### Contributed by  
[!TODO] <REPLACE: Person, or group that provided input or support to the datasets development but may not have been the primary creators. Contributions can include sample collection, processing, analysis, documentation, and-or submission of the dataset. This suggests collaboration, where multiple parties might have played various roles in the dataset development. Can list below> <metadata_key: dataset_contributors> 
  
---  
## Related Resources  
  
### Related datasets, standards, metadata, and ontologies  
[!TODO] <REPLACE: If the dataset is related to or derived from other datasets, standards, metadata and ontologies, please list those datasets and describe the relationship. For example, This dataset was derived from [DATASET NAME] (DOI: [DATASET DOI]) by applying [TRANSFORMATION OR PROCESS].> <metadata_key: related_resources.related_datasets>  
  
### Related publications  
[!TODO] <REPLACE: List any publications that are associated with the dataset, including DOIs, arXiv IDs, or URLs.> <metadata_key: related_resources.publications>  
  
### Related software  
[!TODO] <REPLACE: List any software that is associated with the dataset, including links or PIDs if available.> <metadata_key: related_resources.software>  

### Related ai model   
[!TODO] <REPLACE: List any software that is associated with the dataset, including links or PIDs if available.> <metadata_key: related_resources.aimodels>    

---  
## Methods  
  
### Dataset generation, collection, and procedures  
[!TODO] <REPLACE: Describe how the dataset was generated or collected. For example, raw experimental measurements from user facilities, processed, physics-ready experimental data, outputs from computational simulations, or data derived from prior datasets? For each instrument, facility, or source used to generate and collect the data, what mechanisms or procedures were used for the data collection? If the data was derived, list and describe the source(s) and describe how they were used.>  
  
### Maintenance & Updates  
[!TODO] <REPLACE: Describe update expectations and stewardship responsibility.> <metadata_key: stewardship><metadata_key: maintenance>  
  
  
# ---- Level 3: Understandable & Trustworthy ----  
  
### Data Characteristics  
[!TODO] <REPLACE: Describe variables or features, schema conventions, and missing data handling.><metadata_key: dataset_info.features>  
  
### Data Quality & Limitations  
[!TODO] <REPLACE: Describe completeness, known issues, uncertainties, noise characteristics, and bias considerations.><metadata_key: data_quality>  
  
### Related Schemas or Ontologies  
[!TODO] <REPLACE: list any relevant schemas, ontologies, or vocabularies.><metadata_key: semantic_layer>  
  
### List of variable name(s), description(s), unit(s), and value labels for each variable in the dataset/file.  
[!TODO] <REPLACE: If appropriate, replace the example table with a table listing each variable in the dataset or file, along with its description, unit, and any value labels if applicable.>  
  
For example:  
| Variable Name | Description  | Unit  | Value Labels  |  
|---------------|---------------------------|-----------|-----------------------------|  
| temp  | Temperature measurement  | Celsius  | N/A  |  
| status  | Operational status  | N/A  | 0 = Off, 1 = On  |  
  
### Codes used for missing data  
[!TODO] <REPLACE: Replace the example table of codes used to represent missing data in the dataset or file.>  
  
For example:  
| Code | Description  |  
|------|---------------------------|  
| -999 | Data not collected  |  
| -888 | Measurement error  |  
  
### Specialized formats or other abbreviations used  
[!TODO] <REPLACE: Describe any specialized data formats, abbreviations, or conventions used in the dataset or file. For example, if the dataset is in a specific file format (e.g., ROOT, HDDM, HDF5), or if there are any domain-specific abbreviations used in variable names or values.>  
  
### Example of the contents  
[!TODO] <REPLACE: Optional. Provide a sample of the dataset or file, or a citation (in bibtex format) or link to where one can review an example of the contents. This can help users understand the structure and content of the dataset.>  
  
### Data Processing  
[!TODO] <REPLACE: Describe preprocessing, calibration, filtering, labeling, or transformations applied to the dataset.>  
  
### Software used to preprocess/ clean/ label the data  
[!TODO] <REPLACE: If the software used to preprocess, clean, or label the data is available, please provide a bibtex format, PID, link, or other access point, along with descriptions of any required packages or libraries to run the scripts.> <metadata_key: related_resources.software>  
  
## Integrity & Versioning  
[!TODO] <REPLACE: Describe checksum availability, fixity strategy, and dataset versioning approach.><metadata_key: integrity>  
  
## Semantic / Schema Information  
[!TODO] <REPLACE: Describe schema, ontology alignment, semantic context, and controlled vocabularies. If no formal schema or ontology exists, this section may remain empty.  Examples may include:  JSON Schema or XML schema, NETCDF CF conventions, data dictionary or feature definitions, domain  ontologies like ENVO, controlled vocabularies, or units standards. For example:  schema_URL: "https://example.org/schema.json" or ontology_alignment: "http://purl.obolibrary.org/obo/ENVO_00002005"><metadata_key: semantic_layer>  
  
## AI / Machine Learning Considerations  
[!TODO] <REPLACE: Describe appropriate AI/ML uses, restrictions, bias risks, and safety considerations.><metadata_key: ai_usage>  
  
---  
  
## Additional Information  
[!TODO] <REPLACE: Optional. Include any relevant contextual notes.>  

---
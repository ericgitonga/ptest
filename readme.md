
## Conda Env Install
```shell
% conda env create -f environment.yml
% conda activate pgaff
```

## Pre-commit
Sync your local repo with main to ensure you have the following files in the ecoscope-pgaff root folder:
 - .pre-commit-config.yaml

Run `pip install pre-commit`

Run `pre-commit install`

## Run Pytest

### 1. Environment Setup
Add the following to your `.env` file in the ecoscope-pgaff root folder

``` 

ER_SERVER=
ER_USERNAME=
ER_PASSWORD=

EE_ACCOUNT=
EE_PROJECT=

SM_SERVER=
SM_USERNAME=
SM_PASSWORD=
```

Install Playwright
```shell
% playwright install
```

### 2. Run Tests

```shell
% pytest
```
### 3. Develop Tests
Adding tests to your script is strongly recommended to prevent breakage from future changes. You can find examples in the `tests` directory to help you develop basic smoke tests for your script. Here's a step-by-step guide:

#### File Structure
```python
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent
SCRIPT_PATH = os.path.join(SCRIPT_DIR, <folder_name>, <script_name>)

def test_<script_name>_runs(tmp_path):
    env = os.environ.copy()

    # Add all the environment variables required by your script
    # e.g. env["OUTPUT_DIR"] = tmp_path
    # You don't want to add any credentials here. 
    # Credentials should be set up in your local .env file and in Github Variables
    # EarthRanger, Google Earth Engine and Smart credentials have already been added to Github environment
    env[<VARIABLE_NAME>] = <VARIABLE_VALUE>
    env["OUTPUT_DIR"] = tmp_path

    process = subprocess.run([sys.executable, str(SCRIPT_PATH)], capture_output=True, text=True, env=env)

    assert process.returncode == 0

    # Check if the output has been successfully generated.
    assert os.path.exists(os.path.join(tmp_path, <file_name>))
```

Important Note: 
1. Customize all values in angle brackets (`<>`) according to your specific script.
2. For variables that typically require user input, provide an environment variable fallback using `os.getenv(<VARIABLE_NAME>) or input(...)`
3. Define `OUTPUT_DIR` in your script as a configurable variable. This allows setting it to tmp_path during testing to verify output file generation."


## Migrating from helper functions to workflows tasks
In order to smooth the migration of the scripts in this repository to `ecoscope-workflows` workflow specs, 
we can start making use of tasks defined in the `ecoscope-workflows` library here.
This example specifically steps through updating the `ATE_Analysis.py` script at the time of writing
We're going to update it to use `ecoscope-workflows` tasks for initialising the EarthRanger client and calculating the ETD 

First thing is to import the required tasks from ecoscope workflows
```diff
import ecoscope
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import helper as helper
+from ecoscope_workflows_ext_ecoscope.connections import EarthRangerConnection
+from ecoscope_workflows_ext_ecoscope.tasks.analysis import calculate_time_density
```

The initialisation of an EarthRanger client would now look like this 
Note that the credentials gathering logic has now been surfaced at the top level of the script:
```diff
def main():

    # Initialize EarthRanger
-   er_io = helper.earthranger_init()
+   er_server = os.getenv('ER_SERVER') or input("Enter the EarthRanger server URL (default: https://mep-dev.pamdas.org): ") or "https://mep-dev.pamdas.org"
+   er_username = os.getenv('ER_USERNAME') or input("Enter your EarthRanger username: ")
+   er_password = os.getenv('ER_PASSWORD') or getpass.getpass("Please enter your ER password: ")
    
+   er_io = EarthRangerConnection(
+       server=er_server,
+       username=er_username,
+       password=er_password,
+       tcp_limit=5,
+       sub_page_size=4000
+   ).get_client()
```

We can call the EarthRanger client's `get_subjectgroup_observations` directly
```diff
# Download relocations from ER
-relocs = helper.fetch_relocations(er_io, None, None, subject_group_name)
+relocs = er_io.get_subjectgroup_observations(
+        subject_group_name=subject_group_name,
+        since=None,
+        until=None,
+        include_subject_details=True,
+        include_subjectsource_details=True,
+        include_details=True
+    )
```

We can replace `helper.compute_etd_percentiles` with the workflows task `calculate_time_density`
```diff
# calculate an ETD HR for each subject
-    indv_total_etd = helper.compute_etd_percentiles(traj, 
-                                                    output_dir=output_dir, 
-                                                    desired_percentiles=[desired_percentile],
-                                                    )
+    indv_total_etd = calculate_time_density(traj, desired_percentiles=[desired_percentile])
```

It's worth noting that this is a particularly 'happy' example, the translation of helper function to task was basically 1:1
This won't always be true, and in some cases deeper changes might be required, either to the script or as updates to tasks in `ecoscope-workflows`


## Contents of the ER-DATA-MOVEMENT/.env file: 
ER_SERVER_1=<ER1 URL>
ER_USERNAME_1=<ER1 USERNAME>
ER_PASSWORD_1=<ER1 PASSWORD>
ER_SERVER_2=<ER2 URL>
ER_USERNAME_2=<ER2 USERNAME>
ER_PASSWORD_2=<ER2 PASSWORD>
SUBJECTGROUP_NAME= #None
EVENT_TYPE= #None
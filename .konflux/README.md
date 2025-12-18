The contents of this folder (as well as the Cargo.toml and Cargo.lock in the root directory) exist to 
  solve some issues when doing hermetic builds with Hermeto. 

1. [This bug in hermeto](https://github.com/hermetoproject/hermeto/issues/1205), where sdists are not fetched if a binary wheel is not present for a given cpu architecture. I was able to workaround this by creating fake context folders (`.konflux/s390x` and `.konflux/ppc64le`) and having separate hermeto inputs for those folders.

```
      {
        "type": "pip", 
        "path": ".", 
        "requirements_files": ["requirements-x86_64.txt"],
        "requirements_build_files": ["requirements-build.txt"],
        "binary": { 
          "arch": "x86_64,aarch64"
        }
      },
      {
        "type": "pip", 
        "path": ".konflux/s390x", 
        "requirements_files": ["requirements.txt"],
        "requirements_build_files": ["requirements-build.txt"],
        "binary": { 
          "arch": "s390x"
        }
      },
      {
        "type": "pip", 
        "path": ".konflux/ppc64le", 
        "requirements_files": ["requirements.txt"],
        "requirements_build_files": ["requirements-build.txt"],
        "binary": { 
          "arch": "ppc64le"
        }
      },

```

2. Possibly another bug in hermeto when building rust dependencies of pip packages. Not 100% sure on this one, but I think if a rust dependency's repo does not contain a Cargo.lock file, hermeto won't prefetch it. To workaround this, I did the following:

- Added a `{"type": "cargo"}` block to the hermeto input
- Added the broken dependencies to a root Cargo.toml
- Created a dummy `.konflux/main.rs` file and referenced it in Cargo.toml. 

With that in place, I could run `cargo generate-lockfile` to get a `Cargo.lock` that hermeto would prefetch.


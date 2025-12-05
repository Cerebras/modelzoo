# modelzoo-internal

This is a repo to stage our model zoo and other customer models before releasing them generally, to the public or specific customer shared repos when they are ready.

Each sub directory other than [utils](./utils) correspond to a shared repo. 

Currently, [modelzoo](./modelzoo) is the most frequently updated repo and others are updated on ad-hoc basis as needed.

For [modelzoo](./mnodelzoo), once a day in the evening, we pull code from `src/models` in [monolith](https://github.com/Cerebras/monolith/tree/master/src/models) on a daily basis. When we’re ready to release, we push (read: _copy_) code from MZ-internal release branch to MZ master branch and then tag that commit id with the appropriate release number.
MZ is public and we currently don’t push development code to public repo, hence the need for MZ-internal. Hackathons are held on MZ-internal release branch

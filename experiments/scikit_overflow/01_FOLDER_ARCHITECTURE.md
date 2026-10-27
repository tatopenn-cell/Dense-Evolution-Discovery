# 01_FOLDER_ARCHITECTURE: the folder is the state

This package follows the architecture of the kit: memory, skills, protocols,
harness. The chat is not part of the package.

## The layout

~~~text
scikit_overflow_package/
├── README.md
├── 01_FOLDER_ARCHITECTURE.md
├── ISTRUZIONI.md
├── SKILL.md
├── 04_SELF_CHECK.md
├── 05_DELIVERY_TEMPLATE.md
├── PROMPT.txt
└── papers/
    └── INDEX.md
~~~

## Why the folder

A prompt lives in one string, it is not diffable, it is not reusable. A folder
of files is diffable, reusable, and it can be read in a fixed order.

## The one rule

A file that an item of `ISTRUZIONI.md` does not name does not exist. If the
agent needs it, it is a file; if it is not a file, the agent does not have it.

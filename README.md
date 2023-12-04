# jinspector - explore unknown json data
Empower your JSON data exploration with this inspection tool. Unleash the full potential of your JSON files through seamless schema discovery, precision-driven data inspection using a JSON Path query language, and effortlessly save filtered content to a new file. Elevate your JSON file analysis to new heights, making data exploration an intuitive and efficient process.
## Design choices
- **Minimalist Dependency:** Harness the power of standard Python 3.x modules exclusively. No additional installations required, ensuring a streamlined and hassle-free user experience.

- **Simplicity in Deployment:** Experience effortless setup with this one-file tool. Deployment is as straightforward as copying a single file — a quick and efficient solution for users seeking simplicity in implementation.

- **Command-Line Efficiency:** Embrace the command line for a powerful and efficient experience. This tool forgoes unnecessary graphical interfaces, focusing on console interactions to provide a direct and responsive user interface.

- **Platform-Centric Performance:** Designed with a primary focus on Linux, ensuring optimal performance and compatibility within the Linux environment. Streamlined and tailored for users operating on Linux platforms.

## Description
Experience the flexibility of two distinct modes: command line and shell.

**Command Line Mode:** Execute scripts seamlessly from your shell, allowing for quick and efficient operations. For example:
`$ python3 jinspector.py -s "load my.json | filter '$.table[20:30] | save out.json`

**Shell Mode:** Immerse yourself in an interactive experience where you are prompted to input jinspector commands, providing a dynamic and hands-on approach to exploring and manipulating your JSON data.
```
(jinspect) load 0.json

(jinspect) schema
$: <Object> {
  firstName: <str>
  lastName: <str>
  age: <int>
}
(jinspect) quit
```
## Examples
- `load my.json | schema`
- `load my.json | filter document[?(@.price > 100)]`
- `python3 jinspector.py -s "load 0.json | filter a.arr[1:2]."`
## Running tests
`python3 -m unittest discover`

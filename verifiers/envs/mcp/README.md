# Verifiers MCP Environments

![MCP Environment Overview](../../../../mcp-env-v2/public/mcp-env.png)

The MCP Env abstraction aims to allow easily implementing Verifiers environments that use MCP servers and their corresponding tools instead of manually implementing tools.  MCP itself supports a few different paths for server builders depending on their use case and the MCP env should enable many of these setups.  

The transports that MCP supports include:

- stdio
- http
- sse (depracated)

## Scenarios

- stdio
  - Small scale, stateless, local use cases can be done easily with the stdio transport. Stdio is meant to be a local communication so server and client are ran locally with the server being run on a background process.
- http
  - Any scale, stateless, remote use cases in which the MCP you want to use is provided via some MCP server provider.  In this case instead of running the server via a command, you just have to provide the URL and all rollouts can make their connections to the remote server.
  - example: instead of running your own MCP Server you can use a provider like smithery or many companies run their own remote MCP servers that you can connect to.
- sandbox
  - Any scale, stateful use cases are enabled by the "sandbox" transport option which relies on the Streaming HTTP transport communication in which each sandbox will run its own MCP HTTP Server and each rollout can connect to their corresponding MCP via URL.
  - example: say you have a sandbox setup that includes setting up some filesystem or database.  using the sandbox transport will allow each rollout to have its own sandbox it can connect to via MCP HTTP Server and perform actions that change the state of the sandbox.


## Concerns

Some things to note that MCP based environment implementations are relient on the MCP Server developers implementation.  Some MCP developers have provided ways to run their server with the different methods, some only provide a method to run stdio via uvx/node, and some don't provide either but might require something like cloning a repo and running a file manually.  These all effect whether or not they will work as is in an environment implementation so something to think about.  If you are the MCP server developer yourself obviously you have the control to enable any setup you choose.  

When it comes to running servers via HTTP it is also up to you to handle any authentication concerns as a running server may be available to anyone to connect to.


## TODO

- a stdio-http bridge that will provide a wrapper function in case an MCP server doesn't provide an easy way to run the server in HTTP mode, you can still use the stdio version that is then made available by a HTTP wrapper.

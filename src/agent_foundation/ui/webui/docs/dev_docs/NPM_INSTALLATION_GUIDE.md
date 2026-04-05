# NPM/Yarn Package Installation on Meta Devservers

This guide documents the working solution for installing npm packages on Meta devservers, where standard npm/yarn commands are restricted.

## The Problem

Meta devservers have several restrictions that prevent standard npm/yarn package installation:

1. **npm registry blocked**: Direct access to `registry.npmjs.org` returns 403 Forbidden due to Meta's destination filter
2. **Internal npm registry SSL issues**: Using `www.npm.internalfb.com` causes SSL certificate hostname mismatch errors
3. **Production path restrictions**: Running npm from production system paths triggers "npm direct installs not allowed on Production system paths"

## Working Solution

The following combination of tools and configuration successfully enables package installation:

### Prerequisites

1. **fbsource Yarn binary**: Use the Yarn binary from fbsource instead of system npm
2. **Public yarn registry**: Configure to use the public yarn registry
3. **HTTPS proxy**: Use Meta's forward proxy for external network access

### Step-by-Step Instructions

#### 1. Navigate to your React project directory

```bash
cd /path/to/your/react/project
```

Make sure you're in the directory that contains `package.json`.

#### 2. Create/update `.npmrc` configuration

```bash
echo "registry=https://registry.yarnpkg.com/" > .npmrc
```

This tells yarn to use the public registry instead of attempting to use Meta's internal registry.

#### 3. Set the HTTPS proxy

```bash
export HTTPS_PROXY="http://fwdproxy:8080"
```

This enables network access through Meta's forward proxy.

#### 4. Install packages using fbsource Yarn

**To add a single package:**
```bash
~/fbsource/xplat/third-party/yarn/yarn add <package-name>
```

**To install all dependencies from package.json:**
```bash
~/fbsource/xplat/third-party/yarn/yarn install
```

### Complete Example

Here's a complete example of installing `remark-gfm` package:

```bash
# Navigate to React project
cd /data/users/zgchen/fbsource251217/fbcode/_tony_dev/ScienceModelingTools/tools/ui/chatbot_demo_react/react

# Create .npmrc with public registry
echo "registry=https://registry.yarnpkg.com/" > .npmrc

# Set proxy
export HTTPS_PROXY="http://fwdproxy:8080"

# Install the package
~/fbsource/xplat/third-party/yarn/yarn add remark-gfm
```

## Common Errors and Solutions

### Error: 403 Forbidden from registry

**Cause**: Direct npm access is blocked by Meta's network filters.

**Solution**: Use fbsource Yarn with the public registry configuration as described above.

### Error: SSL WRONG_VERSION_NUMBER or Hostname mismatch

**Cause**: SSL certificate issues with Meta's internal npm registry.

**Solution**: Switch to the public yarn registry (`registry.yarnpkg.com`) and use the HTTPS proxy.

### Error: npm direct installs not allowed on Production system paths

**Cause**: Meta security restrictions prevent direct npm usage in certain paths.

**Solution**: Use the fbsource Yarn binary (`~/fbsource/xplat/third-party/yarn/yarn`) instead of system npm.

### Error: Can't find package.json

**Cause**: Running yarn from the wrong directory.

**Solution**: Ensure you're in the directory that contains `package.json`. For this project:
```bash
cd /path/to/chatbot_demo_react/react  # NOT the parent chatbot_demo_react directory
```

## Key Commands Reference

| Task | Command |
|------|---------|
| Add a package | `~/fbsource/xplat/third-party/yarn/yarn add <package>` |
| Install all deps | `~/fbsource/xplat/third-party/yarn/yarn install` |
| Remove a package | `~/fbsource/xplat/third-party/yarn/yarn remove <package>` |
| Update packages | `~/fbsource/xplat/third-party/yarn/yarn upgrade` |

## Troubleshooting

If the above steps don't work:

1. **Verify proxy is set**: `echo $HTTPS_PROXY` should show `http://fwdproxy:8080`
2. **Verify .npmrc exists**: `cat .npmrc` should show `registry=https://registry.yarnpkg.com/`
3. **Verify correct directory**: `ls package.json` should find the file
4. **Check fbsource path**: Ensure your fbsource checkout path is correct (might be `~/fbsource` or `~/fbsource251217` etc.)

## Notes

- The `.npmrc` file is local to the project and won't affect other projects
- The proxy setting is session-specific; you'll need to re-export it in new terminal sessions
- Yarn will create/update `yarn.lock` file to track exact dependency versions

## Related Documentation

- Meta internal npm docs: https://www.internalfb.com/intern/wiki/Npm_at_Meta/
- Yarn documentation: https://yarnpkg.com/getting-started

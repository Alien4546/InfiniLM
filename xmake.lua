option("songshan-offline")
    set_default(false)
    set_showmenu(true)
    set_description("Use the preinstalled Python and pybind11 headers for the Songshan H3 build")
option_end()

if has_config("songshan-offline") then
    local pybind11_include = os.getenv("SONGSHAN_PYBIND11_INCLUDE")
    local python_include = os.getenv("SONGSHAN_PYTHON_INCLUDE")
    if not pybind11_include or not os.isfile(path.join(pybind11_include, "pybind11", "pybind11.h")) then
        raise("Set SONGSHAN_PYBIND11_INCLUDE to the installed pybind11 include directory")
    end
    if not python_include or not os.isfile(path.join(python_include, "Python.h")) then
        raise("Set SONGSHAN_PYTHON_INCLUDE to the Python include directory")
    end
    add_includedirs(pybind11_include, python_include)
else
    add_requires("pybind11")
end

set_toolchains("gcc")

option("cxx11-abi")
    set_default(nil)
    set_showmenu(true)
    set_description("Set _GLIBCXX_USE_CXX11_ABI to match the installed InfiniCore")
    set_values("0", "1")
option_end()

local cxx11_abi = os.getenv("INFINILM_CXX11_ABI") or get_config("cxx11-abi")
if cxx11_abi and cxx11_abi ~= "" then
    if cxx11_abi ~= "0" and cxx11_abi ~= "1" then
        raise("INFINILM_CXX11_ABI must be 0 or 1")
    end
    add_defines("_GLIBCXX_USE_CXX11_ABI=" .. cxx11_abi)
end

-- Add spdlog from third_party directory
add_includedirs("third_party/spdlog/include")
add_includedirs("third_party/json/single_include/")

target("_infinilm")
    if not has_config("songshan-offline") then
        add_packages("pybind11")
    end
    set_default(false)
    add_rules("python.module", {soabi = true})
    set_languages("cxx17")
    set_kind("shared")

    local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

    -- add_includedirs("csrc", { public = false })
    -- add_includedirs("csrc/pybind11", { public = false })
    add_includedirs(INFINI_ROOT.."/include", { public = true })
    -- spdlog is already included globally via add_includedirs at the top

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infinicore_cpp_api", "infiniop", "infinirt", "infiniccl")

    -- Add C++ sources
    add_files("csrc/**.cpp")
    add_files("csrc/**.cc")

    set_installdir("python/infinilm")
target_end()

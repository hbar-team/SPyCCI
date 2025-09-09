from tempfile import NamedTemporaryFile as tmp

from os import system
from os.path import join, basename

from spycci.core.dependency_finder import locate_vmd


def render_molecule(
    molecule_file: str,
    resolution: int = 4800,
    shadows: bool = True,
    ambientocclusion: bool = True,
    dof: bool = True,
    VMD_PATH: str = None,
) -> None:
    """
    Given the path to a molecule file (e.g. .xyz, .pdb) the function saves a `.bmp` render
    of the molecular structure.

    Arguments
    ---------
    molecule_file: str
        The path to the file encoding the structure of the molecule.
    resolution: int
        The resolution of the output image (default: 4800).
    shadows: bool
        If set to True will enable the vmd shadows option
    ambientocclusion: bool
        If set to True will enable the vmd ambientocclusion option
    dof: bool
        If set to True will enable the vmd dof option
    VMD_PATH: str
        The path to the vmd folder. Is set to None (default), will automatically search vmd
        in the system PATH.
    """
    vmd_root = VMD_PATH if VMD_PATH is not None else locate_vmd().rstrip("/bin/vmd")
    tachyon_path = join(vmd_root, "lib/vmd/tachyon_LINUXAMD64")

    root_name = basename(molecule_file).rsplit('.', 1)[0]

    with tmp(mode="w+", suffix=".vmd") as vmd_script:

        vmd_script.write(f"""
        mol addrep 0
        display projection Orthographic
        display resetview
        mol new {molecule_file}


        axes location Off
        mol color Name
        mol representation CPK 1.000000 0.300000 150.000000 12.000000
        mol selection all
        mol material Opaque
        mol addrep 0

        display resetview
        """
        )

        if shadows:
            vmd_script.write("display shadows on\n")
        if ambientocclusion:
            vmd_script.write("display ambientocclusion on\n")
        if dof:
            vmd_script.write("display dof on\n")

        vmd_script.write(
            f"""
            color Display Background white
            color Name C black
            mol modcolor 0 0 Element
            render Tachyon {root_name}.dat "{tachyon_path}" -fullshade -aasamples 12 %s -format BMP -res {resolution} {resolution} -o {root_name}.bmp
            exit
            """
        )

        vmd_script.seek(0)
        system(f"vmd -dispdev text -e {vmd_script.name}")


def render_fukui_cube(
    cubfile: str,
    isovalue: float = 0.003,
    include_negative: bool = False,
    resolution: int = 4800,
    shadows: bool = True,
    ambientocclusion: bool = True,
    dof: bool = True,
    VMD_PATH: str = None,
) -> None:
    """
    Given the path to a Fukui function cube file saves a `.bmp` render the volumetric Fukui
    function.

    Arguments
    ---------
    cubefile: str
        The path to the `.fukui.cube` file that must be rendered.
    isovalue: float
        The isovalue at which the contour must be plotted (default: 0.003).
    include_negative: bool
        If set to True, will render also the negative part of the Fukui function. (default:
        False)
    resolution: int
        The resolution of the output image (default: 4800).
    shadows: bool
        If set to True will enable the vmd shadows option
    ambientocclusion: bool
        If set to True will enable the vmd ambientocclusion option
    dof: bool
        If set to True will enable the vmd dof option
    VMD_PATH: str
        The path to the vmd folder. Is set to None (default), will automatically search vmd
        in the system PATH.
    """

    vmd_root = VMD_PATH if VMD_PATH is not None else locate_vmd().rstrip("/bin/vmd")
    tachyon_path = join(vmd_root, "lib/vmd/tachyon_LINUXAMD64")

    root_name = basename(cubfile).rstrip(".fukui.cube")

    with tmp(mode="w+") as vmd_script:

        vmd_script.write(
            f"""
            mol addrep 0
            display projection Orthographic
            display resetview
            mol new {cubfile} type {{cube}} first 0 last -1 step 1 waitfor 1 volsets {{0 }}
            animate style Loop
            axes location Off
            mol modstyle 0 0 CPK 1.000000 0.300000 12.000000 12.000000
            mol color Name
            mol representation CPK 1.000000 0.300000 150.000000 12.000000
            mol selection all
            mol material Opaque

            mol addrep 0
            mol modcolor 1 0 ColorID 1
            mol modstyle 1 0 Isosurface {isovalue} 0 0 0 1 1
            mol modmaterial 1 0 Translucent
            mol scaleminmax 0 1 0.000000 1.000000
            
            display cuemode Linear
            """
        )

        if include_negative:
            vmd_script.write(
                f"""
                mol addrep 0
                mol modcolor 2 0 ColorID 0
                mol modstyle 2 0 Isosurface {-isovalue} 0 0 0 1 1
                mol modmaterial 2 0 Translucent
                mol scaleminmax 0 2 -1.000000 0.000000
                """
            )

        if shadows:
            vmd_script.write("display shadows on\n")

        if ambientocclusion:
            vmd_script.write("display ambientocclusion on\n")

        if dof:
            vmd_script.write("display dof on")

        vmd_script.write(
            f"""
            color Display Background white
            color Element C black
            mol modcolor 0 0 Element
            render Tachyon {root_name}.dat "{tachyon_path}" -fullshade -aasamples 12 %s -format BMP -res {resolution} {resolution} -o {root_name}.bmp
            exit
            """
        )

        vmd_script.seek(0)
        system(f"vmd -dispdev text -e {vmd_script.name}")


def render_condensed_fukui(
    cubfile: str,
    resolution: int = 4800,
    shadows: bool = True,
    ambientocclusion: bool = True,
    dof: bool = True,
    VMD_PATH: str = None,
) -> None:
    """
    Given the path to a Fukui function cube file saves a `.bmp` render of the condensed Fukui
    functions.

    Arguments
    ---------
    cubefile: str
        The path to the `.fukui.cube` file that must be rendered.
    resolution: int
        The resolution of the output image (default: 4800).
    shadows: bool
        If set to True will enable the vmd shadows option
    ambientocclusion: bool
        If set to True will enable the vmd ambientocclusion option
    dof: bool
        If set to True will enable the vmd dof option
    VMD_PATH: str
        The path to the vmd folder. Is set to None (default), will automatically search vmd
        in the system PATH.
    """
    vmd_root = VMD_PATH if VMD_PATH is not None else locate_vmd().rstrip("/bin/vmd")
    tachyon_path = join(vmd_root, "lib/vmd/tachyon_LINUXAMD64")

    root_name = basename(cubfile).rstrip(".fukui.cube")

    with tmp(mode="w+") as vmd_script:

        vmd_script.write(
            f"""
            mol addrep 0
            display projection Orthographic
            mol new {cubfile} type {{cube}} first 0 last -1 step 1 waitfor 1 volsets {{0 }}
            """
        )

        vmd_script.write(
            """
            animate style Loop
            mol selection all
            mol color Charge
            mol representation Licorice 0.1 20.000000 20.000000
            mol material Opaque
            mol modrep 0 0
            color Display Background white
            color scale method BWR
            display cuemode Linear
            axes location Off

            label delete Atoms all
            set all [atomselect 0 "all"]
            set i 0
            foreach atom [$all list] {
            label add Atoms "0/$atom"
            label textformat Atoms $i {%q}
            label textoffset Atoms $i { 0.025  0.0  }
            incr i
            }

            label textsize 1
            label textthickness 3
            color Labels Atoms black
            display resetview
            """
        )

        if shadows:
            vmd_script.write("display shadows on\n")

        if ambientocclusion:
            vmd_script.write("display ambientocclusion on\n")

        if dof:
            vmd_script.write("display dof on")

        vmd_script.write(
            f"""
            render Tachyon {root_name}_condensed.dat "{tachyon_path}" -fullshade -aasamples 12 %s -format BMP -res {resolution} {resolution} -o {root_name}_condensed.bmp
            exit
            """
        )

        vmd_script.seek(0)
        system(f"vmd -dispdev text -e {vmd_script.name}")


def render_spin_density_cube(
    cubfile: str,
    isovalue: float = 0.005,
    resolution: int = 4800,
    shadows: bool = True,
    ambientocclusion: bool = True,
    dof: bool = True,
    VMD_PATH: str = None,
) -> None:
    """
    Given the path to a spin density cube file saves a `.bmp` render the function.

    Arguments
    ---------
    cubefile: str
        The path to the `.fukui.cube` file that must be rendered.
    isovalue: float
        The isovalue at which the contour must be plotted (default: 0.003).
    resolution: int
        The resolution of the output image (default: 4800).
    shadows: bool
        If set to True will enable the vmd shadows option
    ambientocclusion: bool
        If set to True will enable the vmd ambientocclusion option
    dof: bool
        If set to True will enable the vmd dof option
    VMD_PATH: str
        The path to the vmd folder. Is set to None (default), will automatically search vmd
        in the system PATH.
    """
    vmd_root = VMD_PATH if VMD_PATH is not None else locate_vmd().rstrip("/bin/vmd")
    tachyon_path = join(vmd_root, "lib/vmd/tachyon_LINUXAMD64")

    root_name = basename(cubfile).rstrip(".fukui.cube")

    with tmp(mode="w+") as vmd_script:

        vmd_script.write(
            f"""
            mol addrep 0
            display projection Orthographic
            display resetview
            mol new {cubfile} type {{cube}} first 0 last -1 step 1 waitfor 1 volsets {{0 }}
            animate style Loop
            axes location Off
            mol modstyle 0 0 CPK 1.000000 0.300000 12.000000 12.000000
            mol color Name
            mol representation CPK 1.000000 0.300000 150.000000 12.000000
            mol selection all
            mol material Opaque
            mol addrep 0
            mol modcolor 1 0 ColorID 31
            mol modstyle 1 0 Isosurface {isovalue} 0 0 0 1 1
            mol modmaterial 1 0 Translucent
            mol scaleminmax 0 1 0.000000 1.000000
            mol addrep 0
            mol modcolor 2 0 ColorID 26
            mol modstyle 2 0 Isosurface {-isovalue} 0 0 0 1 1
            mol modmaterial 2 0 Translucent
            mol scaleminmax 0 2 -1.000000 0.000000
            display cuemode Linear
            """
        )

        if shadows:
            vmd_script.write("display shadows on\n")

        if ambientocclusion:
            vmd_script.write("display ambientocclusion on\n")

        if dof:
            vmd_script.write("display dof on")

        vmd_script.write(
            f"""
            color Display Background white
            color Element C black
            mol modcolor 0 0 Element
            render Tachyon {root_name}.dat "{tachyon_path}" -fullshade -aasamples 12 %s -format BMP -res {resolution} {resolution} -o {root_name}.bmp
            exit
            """
        )

        vmd_script.seek(0)
        system(f"vmd -dispdev text -e {vmd_script.name}")

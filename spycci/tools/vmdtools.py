from tempfile import NamedTemporaryFile as tmp

from os import system
from os.path import join, basename, isfile
from typing import List, Optional

from spycci.core.dependency_finder import locate_vmd


class VMDRenderer:
    """
    The `VMDRenderer` class is a simple wrapper developed around the Visual Molecular Dynamics (VMD)
    software. The class allows the user to easily generate images and renders of molecules and cube files.
    Once an instance of the `VMDRenderer` is created, the user can use its render functionality through
    the provided built-in methods.

    Arguments
    ---------
    resolution: int
        The resolution of the output image (default: 800).
    scale: float
        Scales the frame zoom according to the user specified factor (default: 1. no
        zoom is applied)
    xyx_rotation: List[float]
        The list of 3 rotation angles (from 0 to 360°) defyning subsequent rotations around
        the X, Y and X axis. (default: [0., 0., 0.] no rotation is applied).
    shadows: bool
        If set to `True` will enable the vmd shadows option
    ambientocclusion: bool
        If set to `True` will enable the vmd ambientocclusion option
    dof: bool
        If set to `True` will enable the vmd dof option
    VMD_PATH: str
        The path to the vmd executable. Is set to `None` (default), will automatically search `vmd`
        in the system PATH.

    Raises
    ------
    FileNotFoundError
        Exception raised if the user provided `VMD_PATH` is invalid.
    RuntimeError
        Exception raised if either the `vmd` program or the Tachyon ray tracer are not found.
    ValueError
        Exception raised if scale or xyx rotation angles are not properly formatted.
    """

    def __init__(
        self,
        resolution: int = 800,
        scale: float = 1.0,
        xyx_rotation: List[float] = [0.0, 0.0, 0.0],
        shadows: bool = True,
        ambientocclusion: bool = True,
        dof: bool = True,
        VMD_PATH: Optional[str] = None,
    ) -> None:

        if scale <= 0.0:
            raise ValueError(
                "The VMD frame scale factor must be a non-zero positive float."
            )
        self.__scale: float = scale

        if len(xyx_rotation) != 3:
            raise ValueError("The XYX rotation must be a list of 3 float values.")
        self.__xyx_rotation: float = [float(v) for v in xyx_rotation]

        self.resolution: int = resolution
        self.shadows: bool = shadows
        self.ambientocclusion: bool = ambientocclusion
        self.dof: bool = dof

        self.__vmd_root = None
        if VMD_PATH and isfile(VMD_PATH) is False:
            raise FileNotFoundError(
                f"The VMD_PATH {VMD_PATH} does not point to a valid vmd executable."
            )

        elif VMD_PATH:
            self.__vmd_root = locate_vmd(VMD_PATH).rstrip("/bin/vmd")

        else:
            self.__vmd_root = locate_vmd().rstrip("/bin/vmd")

        self.__tachyon_path = join(self.__vmd_root, "lib/vmd/tachyon_LINUXAMD64")
        if isfile(self.__tachyon_path) is False:
            raise RuntimeError(
                f"Cannot locate the Tachyon ray tracer (required by VMD)."
            )

    @property
    def scale(self) -> float:
        """
        The float value setting the frame zoom. The value 1. is set by VMD on startup
        to render the whole molecule in the unrotated frame.

        Return
        ------
        float
            The scale of the frame.
        """
        return self.__scale

    @scale.setter
    def scale(self, value: float) -> None:
        if value <= 0.0:
            raise ValueError(
                "The VMD frame scale factor must be a non-zero positive float."
            )
        self.__scale: float = value

    @property
    def xyx_rotation(self) -> List[float]:
        """
        The list of 3 rotation angles (from 0 to 360°) defyning subsequent rotations around
        the X, Y and X axis.

        Return
        ------
        List[float]
            The list of XYX rotation angles.
        """
        return self.__xyx_rotation

    @xyx_rotation.setter
    def xyx_rotation(self, value: List[float]) -> None:
        if len(value) != 3:
            raise ValueError("The XYX rotation must be a list of 3 float values.")
        self.__xyx_rotation: float = [float(v) for v in value]

    def _render(
        self,
        instructions: str,
        output_name: str,
    ) -> None:
        """
        Given a set of `vmd` instructions, run a render operation outputting a `.bmp`
        image file.

        Arguments
        ---------
        instructions: str
            The string encoding the operations to be executed by `vmd`.
        output_name: str
            The name of the output file (`output_name.bmp`).
        """

        with tmp(mode="w+", suffix=".vmd") as vmd_script:

            vmd_script.write(instructions)

            vmd_script.write(f"rotate x by {self.__xyx_rotation[0]}\n")
            vmd_script.write(f"rotate y by {self.__xyx_rotation[1]}\n")
            vmd_script.write(f"rotate x by {self.__xyx_rotation[2]}\n")
            vmd_script.write(f"scale by {self.__scale}\n")

            if self.shadows:
                vmd_script.write("display shadows on\n")
            if self.ambientocclusion:
                vmd_script.write("display ambientocclusion on\n")
            if self.dof:
                vmd_script.write("display dof on\n")

            vmd_script.write(
                f"""
                render Tachyon {output_name}.dat "{self.__tachyon_path}" -fullshade -aasamples 12 %s -format BMP -res {self.resolution} {self.resolution} -o {output_name}.bmp
                exit
                """
            )

            vmd_script.seek(0)
            system(f"vmd -dispdev text -e {vmd_script.name}")

    def render_molecule(self, molecule_file: str) -> None:
        """
        Given the path to a molecule file (e.g. .xyz, .pdb) the function saves a `.bmp` render
        of the molecular structure.

        Arguments
        ---------
        molecule_file: str
            The path to the file encoding the structure of the molecule.
        """
        root_name = basename(molecule_file).rsplit(".", 1)[0]

        script = ""
        script += "display projection Orthographic\n"
        script += "display resetview\n"
        script += "axes location Off\n"
        script += "color Display Background white\n"
        script += "color Name C black\n"

        script += f"mol new {molecule_file}\n"
        script += "mol addrep 0\n"
        script += "mol color Name\n"
        script += "mol representation CPK 1.000000 0.300000 150.000000 12.000000\n"
        script += "mol selection all\n"
        script += "mol material Opaque\n"
        script += "mol modcolor 0 0 Element\n"

        self._render(script, root_name)

    def render_fukui_cube(
        self, cubfile: str, isovalue: float = 0.003, include_negative: bool = False
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
        """

        root_name = basename(cubfile).rstrip(".fukui.cube")

        script = ""
        script += "display projection Orthographic\n"
        script += "display resetview\n"
        script += "axes location Off\n"
        script += "color Display Background white\n"
        script += "color Name C black\n"

        script += f"mol new {cubfile} type {{cube}} first 0 last -1 step 1 waitfor 1 volsets {{0 }}\n"
        script += "mol modstyle 0 0 CPK 1.000000 0.300000 150.000000 12.000000\n"
        script += "mol color Name\n"
        script += "mol selection all\n"
        script += "mol material Opaque\n"
        script += "mol addrep 0\n"

        script += "mol modcolor 1 0 ColorID 1\n"
        script += f"mol modstyle 1 0 Isosurface {isovalue} 0 0 0 1 1\n"
        script += "mol modmaterial 1 0 Translucent\n"
        script += "mol scaleminmax 0 1 0.000000 1.000000\n"
        script += "display cuemode Linear\n"

        if include_negative:
            script += "mol addrep 0\n"
            script += "mol modcolor 2 0 ColorID 0\n"
            script += "mol modstyle 2 0 Isosurface {-isovalue} 0 0 0 1 1\n"
            script += "mol modmaterial 2 0 Translucent\n"
            script += "mol scaleminmax 0 2 -1.000000 0.000000\n"
                

        self._render(script, root_name)

    def render_condensed_fukui(self, cubfile: str) -> None:
        """
        Given the path to a Fukui function cube file saves a `.bmp` render of the condensed Fukui
        functions.

        Arguments
        ---------
        cubefile: str
            The path to the `.fukui.cube` file that must be rendered.
        """
        root_name = basename(cubfile).rstrip(".fukui.cube")

        script = ""
        script += "display projection Orthographic\n"
        script += "display resetview\n"
        script += "axes location Off\n"
        script += "color Display Background white\n"
        script += "color Name C black\n"

        script += f"mol new {cubfile} type {{cube}} first 0 last -1 step 1 waitfor 1 volsets {{0 }}\n"
        script += "mol addrep 0\n"
        script += "mol modstyle 0 0 Licorice 0.1 20.000000 20.000000\n"
        script += "color scale method BWR\n"
        script += "mol modcolor 0 0 Charge\n"
        script += "mol color Charge\n"
        
        script += "mol selection all\n"
        script += "mol material Opaque\n"
        script += "display cuemode Linear\n"

        script += "label delete Atoms all\n"
        script += """set all [atomselect 0 "all"]\n"""
        script += "set i 0\n"
        script += "foreach atom [$all list] {\n"
        script += """    label add Atoms "0/$atom"\n"""
        script += """    label textformat Atoms $i {%q}\n"""
        script += """    label textoffset Atoms $i { 0.025  0.0  }\n"""
        script += """    incr i\n"""
        script += "}\n"
        script += "label textsize 1.5\n"
        script += "label textthickness 3\n"
        script += "color Labels Atoms black\n"
            
        self._render(script, root_name)

    def render_spin_density_cube(self, cubfile: str, isovalue: float = 0.005) -> None:
        """
        Given the path to a spin density cube file saves a `.bmp` render the function.

        Arguments
        ---------
        cubefile: str
            The path to the `.fukui.cube` file that must be rendered.
        isovalue: float
            The isovalue at which the contour must be plotted (default: 0.003).
        xyx_rotation: Optional[tuple]
            The tuple of 3 rotation angles (from 0 to 360°) defyning subsequent rotations around
            the X, Y and X axis. If None (default), no rotation is applied.
        """

        root_name = basename(cubfile).rstrip(".spindens.cube")

        script = ""
        script += "display projection Orthographic\n"
        script += "display resetview\n"
        script += "axes location Off\n"
        script += "color Display Background white\n"
        script += "color Name C black\n"

        script += f"mol new {cubfile} type {{cube}} first 0 last -1 step 1 waitfor 1 volsets {{0 }}"
        script += "mol addrep 0"
        script += "mol modstyle 0 0 CPK 1.000000 0.300000 150.000000 12.000000"
        script += "mol color Name"

        script += "mol selection all"
        script += "mol material Opaque"
        
        script += "mol modcolor 1 0 ColorID 31"
        script += f"mol modstyle 1 0 Isosurface {isovalue} 0 0 0 1 1"
        script += "mol modmaterial 1 0 Translucent"
        script += "mol scaleminmax 0 1 0.000000 1.000000"

        script += "mol modcolor 2 0 ColorID 26"
        script += "mol modstyle 2 0 Isosurface {-isovalue} 0 0 0 1 1"
        script += "mol modmaterial 2 0 Translucent"
        script += "mol scaleminmax 0 2 -1.000000 0.000000"

        script += "display cuemode Linear"

        self._render(script, root_name)

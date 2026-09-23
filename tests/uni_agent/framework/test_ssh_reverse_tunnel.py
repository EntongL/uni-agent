import pytest

from uni_agent.framework.ssh_reverse_tunnel import (
    SshReverseTunnelConfig,
    build_ssh_reverse_tunnel_command,
)


@pytest.mark.cpu
@pytest.mark.level0
def test_build_ssh_reverse_tunnel_command_requests_dynamic_remote_port():
    config = SshReverseTunnelConfig(
        ssh_host="atlas.example",
        ssh_user="root",
        identity_file="/root/.ssh/id_ed25519",
        known_hosts_file="/root/.ssh/known_hosts",
    )

    command = build_ssh_reverse_tunnel_command(
        config,
        "http://172.16.54.72:42259/sessions/session-1/v1",
    )

    assert command[:4] == ["ssh", "-N", "-T", "-v"]
    assert "-R" in command
    assert command[command.index("-R") + 1] == "127.0.0.1:0:172.16.54.72:42259"
    assert command[-1] == "root@atlas.example"
    assert ["-i", "/root/.ssh/id_ed25519"] == command[command.index("-i") : command.index("-i") + 2]


@pytest.mark.cpu
@pytest.mark.level0
def test_build_ssh_reverse_tunnel_command_formats_ipv6_gateway():
    config = SshReverseTunnelConfig(ssh_host="atlas.example", ssh_user="root", remote_port=18080)

    command = build_ssh_reverse_tunnel_command(config, "http://[2001:db8::72]:42259/sessions/s/v1")

    assert command[command.index("-R") + 1] == "127.0.0.1:18080:[2001:db8::72]:42259"


@pytest.mark.cpu
@pytest.mark.level0
def test_tunnel_config_requires_ssh_host_and_user():
    with pytest.raises(ValueError, match="ssh_reverse_tunnel.ssh_host"):
        SshReverseTunnelConfig.from_mapping({"ssh_user": "root"})
    with pytest.raises(ValueError, match="ssh_reverse_tunnel.ssh_user"):
        SshReverseTunnelConfig.from_mapping({"ssh_host": "atlas.example"})
